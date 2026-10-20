"""Controlled modular-addition training. No topology runs in the training loop."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from .config import TopologyStudyConfig


class StudyTransformer(nn.Module):
    """One causal, bias-free attention/MLP block with two residual additions."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.embedding = nn.Embedding(config.modulus + 1, d)
        self.position = nn.Parameter(torch.empty(3, d))
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attention_out = nn.Linear(d, d, bias=False)
        self.mlp_in = nn.Linear(d, config.d_ff, bias=False)
        self.mlp_out = nn.Linear(config.d_ff, d, bias=False)
        self.unembed = nn.Linear(d, config.modulus, bias=False)
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=config.init_std)

    def representations(self, inputs):
        c = self.config
        x = self.embedding(inputs) + self.position
        q, k, v = [t.reshape(-1, 3, c.n_heads, c.d_model // c.n_heads).transpose(1, 2)
                   for t in self.qkv(x).chunk(3, dim=-1)]
        scores = q @ k.transpose(-1, -2) / (c.d_model // c.n_heads) ** 0.5
        mask = torch.ones(3, 3, dtype=torch.bool, device=x.device).triu(1)
        attention = scores.masked_fill(mask, -torch.inf).softmax(-1) @ v
        x = x + self.attention_out(attention.transpose(1, 2).reshape(-1, 3, c.d_model))
        mlp = self.mlp_in(x).relu()
        residual = x + self.mlp_out(mlp)
        return {'post_block_residual': residual[:, 2], 'mlp_activation': mlp[:, 2]}

    def forward(self, inputs):
        return self.unembed(self.representations(inputs)['post_block_residual'])


def build_data(config):
    pairs = torch.cartesian_prod(torch.arange(config.modulus), torch.arange(config.modulus))
    inputs = torch.cat((pairs, torch.full((len(pairs), 1), config.modulus)), dim=1)
    labels = pairs.sum(1) % config.modulus
    split_generator = torch.Generator().manual_seed(config.split_seed)
    order = torch.randperm(len(pairs), generator=split_generator)
    n = int(len(pairs) * config.train_fraction)
    probe_generator = torch.Generator().manual_seed(config.probe_seed)
    probes = torch.randperm(len(pairs), generator=probe_generator)[:config.probe_size]
    return inputs, labels, {'train': order[:n], 'heldout': order[n:], 'probes': probes,
                            'split_generator': split_generator.get_state(),
                            'probe_generator': probe_generator.get_state()}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def provenance():
    def git(*args):
        return subprocess.check_output(['git', *args], text=True).strip()
    try:
        commit, dirty = git('rev-parse', 'HEAD'), bool(git('status', '--porcelain'))
        diff = git('diff', 'HEAD')
    except (OSError, subprocess.CalledProcessError):
        commit, dirty, diff = None, None, None
    return {'commit': commit, 'dirty': dirty, 'tracked_diff': diff,
            'source_files': {p.name: p.read_text(encoding='utf-8') for p in
                             [Path(__file__), Path(__file__).with_name('config.py')]},
            'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
            'python': sys.version}


def load_checkpoint(path):
    # Checkpoints contain Python/NumPy RNG state. Load only trusted local files.
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.no_grad()
def evaluate(model, inputs, labels, indices, batch_size, device):
    model.eval()
    loss, correct = 0., 0
    for batch in indices.split(batch_size):
        target = labels[batch].to(device)
        logits = model(inputs[batch].to(device))
        loss += nn.functional.cross_entropy(logits.double(), target, reduction='sum').item()
        correct += (logits.argmax(-1) == target).sum().item()
    return loss / len(indices), correct / len(indices)


def run(config, output_dir, *, device='cpu', resume=None, stop_at=None):
    """stop_at supports interruption tests without changing the planned budget."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise ValueError('new training requires an empty output directory')
    torch.use_deterministic_algorithms(True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    inputs, labels, indices = build_data(config)
    model = StudyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay,
                                 betas=(config.beta1, config.beta2), eps=config.adam_eps,
                                 foreach=False)
    training_generator = torch.Generator().manual_seed(config.seed + 1)
    history, start, source = [], 0, provenance()
    if resume:
        saved = load_checkpoint(resume)
        if saved['config'] != asdict(config):
            raise ValueError('resume configuration must match the checkpoint')
        if saved['source']['source_files'] != source['source_files']:
            raise ValueError('training/config source changed; restore the saved source before resuming')
        if saved['device'] != str(device):
            raise ValueError('resume on the original device for trajectory equivalence')
        if saved['torch_threads'] != torch.get_num_threads():
            raise ValueError('resume with the original Torch thread count')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        indices, history, start = saved['indices'], saved['history'], saved['step']
        source = saved['source']
        training_generator.set_state(saved['training_generator'])
        restore_rng(saved['rng'])
        if any(int(p.stem.split('_')[-1]) > start for p in output.glob('checkpoint_*.pt')):
            raise ValueError('resume destination contains later checkpoints; use a fresh directory')
    end = config.steps if stop_at is None else min(config.steps, stop_at)
    if end < start:
        raise ValueError('stop step precedes resume checkpoint')

    def save(step):
        state = dict(config=asdict(config), model=model.state_dict(), optimizer=optimizer.state_dict(),
                     indices=indices, rng=rng_state(), step=step, history=history,
                     source=source, device=str(device), training_generator=training_generator.get_state())
        state['torch_threads'] = torch.get_num_threads()
        path = output / f'checkpoint_{step:08d}.pt'
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(path)
        (output / 'training.json').write_text(json.dumps(history, indent=2))

    def record(step):
        row = {'step': step}
        for name in ('train', 'heldout'):
            loss, accuracy = evaluate(model, inputs, labels, indices[name], config.evaluation_batch_size, device)
            row.update({f'{name}_loss': loss, f'{name}_accuracy': accuracy})
        row['parameter_l2'] = sum(p.detach().double().square().sum().item() for p in model.parameters()) ** .5
        history.append(row)
        print(json.dumps(row), flush=True)

    (output / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    if not resume:
        record(0)
        save(0)
    train_x, train_y = inputs[indices['train']].to(device), labels[indices['train']].to(device)
    for step in range(start + 1, end + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(train_x).double(), train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite training loss at step {step}')
        loss.backward()
        optimizer.step()
        if step % config.evaluation_interval == 0 or step == config.steps:
            record(step)
        if step % config.checkpoint_interval == 0 or step == end:
            save(step)
    return model, history


def estimate_storage(config):
    with torch.random.fork_rng():
        model = StudyTransformer(config)
    parameters = sum(p.numel() for p in model.parameters())
    # Adam moments + weights, split/probe tensors; RNG/source/zip overhead allowance.
    per = parameters * 12 + config.modulus ** 2 * 8 + config.probe_size * 8 + 100_000
    count = 1 + (config.steps + config.checkpoint_interval - 1) // config.checkpoint_interval
    # Histories are repeated in each checkpoint, so include an upper bound.
    per += (config.steps // config.evaluation_interval + 2) * 400
    return {'parameters': parameters, 'checkpoint_count': count,
            'estimated_checkpoint_bytes_upper': per, 'estimated_total_bytes_upper': per * count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'estimate'])
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/topology_pilot'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--stop-at', type=int)
    defaults = TopologyStudyConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        parser.add_argument('--' + field.name.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args(argv)
    config = TopologyStudyConfig(**{f.name: getattr(args, f.name) for f in fields(defaults)})
    torch.set_num_threads(args.threads)
    print(json.dumps(estimate_storage(config), indent=2))
    if args.command == 'train':
        run(config, args.output_dir, device=args.device, resume=args.resume, stop_at=args.stop_at)


if __name__ == '__main__':
    main()
