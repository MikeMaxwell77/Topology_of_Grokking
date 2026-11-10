"""Experiment 3: temporary fixed translations during branched training."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import TopologyStudyConfig
from .interventions import control_delta, displacement, fit_clusters, geometry
from .study import (StudyTransformer, build_data, load_checkpoint, provenance,
                    restore_rng, rng_state)


def branch(checkpoint, output_dir, *, duration, strength=.5, clusters=8,
           repeats=5, seed=601, device='cpu'):
    saved = load_checkpoint(checkpoint)
    config = TopologyStudyConfig(**saved['config'])
    start = saved['step']
    if duration <= 0 or start + duration >= config.steps:
        raise ValueError('duration must leave a clean recovery period within the saved budget')
    if not 0 <= strength <= 1 or repeats < 1:
        raise ValueError('strength in [0,1] and positive repeats required')
    if saved['device'] != device or saved['torch_threads'] != torch.get_num_threads():
        raise ValueError('use the original training device and Torch thread count')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    x, y, _ = build_data(config)
    indices = saved['indices']
    model = StudyTransformer(config).to(device)
    model.load_state_dict(saved['model'])
    with torch.no_grad():
        z = torch.cat([model.representations(b.to(device))['post_block_residual'].cpu()
                       for b in x.split(config.evaluation_batch_size)])
    train = z[indices['train']]
    centers = fit_clusters(train, clusters, seed)
    delta, groups = displacement(z, centers, train.mean(0), strength)
    conditions = [('clean', torch.zeros_like(delta)), ('targeted', delta)]
    conditions += [(f'{kind}_{r}', control_delta(delta, kind, seed + 1 + r))
                   for kind in ('random', 'rotated') for r in range(repeats)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(checkpoint=str(Path(checkpoint).resolve()), start=start,
                    removal_step=start + duration, duration=duration, strength=strength,
                    clusters=clusters, repeats=repeats, seed=seed, source=provenance(),
                    runner_source=Path(__file__).read_text(),
                    rule='fixed per-example translations; identity backward derivative')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    torch.save(dict(centers=centers, groups=groups, conditions=conditions,
                    indices=indices), output / 'interventions.pt')
    for name, change in conditions:
        arm = output / name
        arm.mkdir()
        model = StudyTransformer(config).to(device)
        model.load_state_dict(saved['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay,
                                     betas=(config.beta1, config.beta2),
                                     eps=config.adam_eps, foreach=False)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        restore_rng(saved['rng'])
        change = change.to(device)
        history = []

        @torch.no_grad()
        def record(step):
            model.eval()
            rows = []
            for split in ('train', 'heldout'):
                idx = indices[split]
                for mode in ('clean', 'active'):
                    loss, correct, clouds = 0., 0, []
                    for batch in idx.split(config.evaluation_batch_size):
                        residual = model.representations(x[batch].to(device))['post_block_residual']
                        if mode == 'active':
                            residual = residual + change[batch]
                        logits = model.unembed(residual).double()
                        target = y[batch].to(device)
                        loss += F.cross_entropy(logits, target, reduction='sum').item()
                        correct += (logits.argmax(1) == target).sum().item()
                        clouds.append(residual.cpu())
                    rows.append(dict(step=step, split=split, mode=mode,
                                     loss=loss / len(idx), accuracy=correct / len(idx),
                                     intervention_on_next_update=step < start + duration,
                                     **geometry(torch.cat(clouds), groups[idx])))
            history.extend(rows)
            (arm / 'training.json').write_text(json.dumps(history, indent=2))
            print(json.dumps(dict(arm=name, step=step, measurements=rows)), flush=True)

        def save(step):
            state = dict(config=saved['config'], model=model.state_dict(),
                         optimizer=optimizer.state_dict(), step=step, indices=indices,
                         rng=rng_state(), history=history, source=protocol['source'],
                         device=device, torch_threads=torch.get_num_threads(),
                         training_generator=saved['training_generator'],
                         branch_protocol=protocol, arm=name)
            path = arm / f'checkpoint_{step:08d}.pt'
            temporary = path.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(path)

        record(start)
        save(start)
        train_idx = indices['train']
        train_x, train_y = x[train_idx].to(device), y[train_idx].to(device)
        for step in range(start + 1, config.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            residual = model.representations(train_x)['post_block_residual']
            if step <= start + duration:
                residual = residual + change[train_idx]
            loss = F.cross_entropy(model.unembed(residual).double(), train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss: {name}, step {step}')
            loss.backward()
            optimizer.step()
            boundary = step in (start + duration, config.steps)
            if step % config.evaluation_interval == 0 or boundary:
                record(step)
            if step % config.checkpoint_interval == 0 or boundary:
                save(step)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--duration', required=True, type=int)
    parser.add_argument('--strength', type=float, default=.5)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=601)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    args = vars(parser.parse_args())
    torch.set_num_threads(args.pop('threads'))
    branch(**args)


if __name__ == '__main__':
    main()
