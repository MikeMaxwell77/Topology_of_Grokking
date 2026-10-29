"""Offline cluster-center intervention sweep for trusted study checkpoints."""
import argparse
import glob
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import TopologyStudyConfig
from .study import StudyTransformer, build_data, load_checkpoint


def fit_clusters(z, k, seed=601, iterations=100):
    """Training-only Lloyd clustering; deterministic local RNG, no labels."""
    if not 1 < k <= len(z):
        raise ValueError('need 1 < clusters <= training sample count')
    generator = torch.Generator().manual_seed(seed)
    centers = z[torch.randperm(len(z), generator=generator)[:k]].clone()
    for _ in range(iterations):
        assignments = torch.cdist(z, centers).argmin(1)
        updated = centers.clone()
        for group in range(k):
            selected = z[assignments == group]
            if len(selected):
                updated[group] = selected.mean(0)
        if torch.allclose(updated, centers, atol=1e-7, rtol=0):
            break
        centers = updated
    return centers


def displacement(z, centers, mean, strength):
    assignments = torch.cdist(z, centers).argmin(1)
    return -strength * (centers[assignments] - mean), assignments


def control_delta(delta, kind, seed):
    generator = torch.Generator().manual_seed(seed)
    if kind == 'random':
        directions = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
        return F.normalize(directions, dim=1) * delta.norm(dim=1, keepdim=True)
    if kind == 'rotated':
        matrix = torch.randn((delta.shape[1], delta.shape[1]), generator=generator,
                             dtype=delta.dtype)
        q, r = torch.linalg.qr(matrix)
        q = q * torch.where(r.diag() < 0, -1., 1.)
        return delta @ q
    raise ValueError(kind)


def geometry(z, groups):
    """Scatter uses original assignments, never reclusters after intervention."""
    mean = z.mean(0)
    total = (z - mean).square().sum().item()
    within = 0.
    for group in groups.unique():
        selected = z[groups == group]
        within += (selected - selected.mean(0)).square().sum().item()
    return {'between_fraction': max(0., 1 - within / total) if total > 0 else None,
            'centered_rms': (total / len(z)) ** .5,
            'activation_rms': z.square().sum(1).mean().sqrt().item()}


@torch.no_grad()
def sweep(checkpoints, output_dir, clusters=8, strengths=(0., .25, .5, 1.),
          repeats=5, seed=601):
    if repeats < 1 or any(not 0 <= s <= 1 for s in strengths):
        raise ValueError('positive repeats and strengths in [0, 1] required')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in checkpoints:
        saved = load_checkpoint(path)
        config = TopologyStudyConfig(**saved['config'])
        model = StudyTransformer(config).eval()
        model.load_state_dict(saved['model'])
        x, y, _ = build_data(config)
        indices = saved['indices']
        z = torch.cat([model.representations(batch)['post_block_residual']
                       for batch in x.split(1024)])
        train = z[indices['train']]
        centers = fit_clusters(train, clusters, seed)
        mean = train.mean(0)
        torch.save({'centers': centers, 'mean': mean, 'train_indices': indices['train'],
                    'checkpoint': str(Path(path).resolve())},
                   output / f'clusters_{len(rows):08d}.pt')
        for strength in strengths:
            delta, groups = displacement(z, centers, mean, strength)
            conditions = [('targeted', -1, delta)]
            if strength:
                conditions += [(kind, repeat, control_delta(delta, kind, seed + 1 + repeat))
                               for kind in ('random', 'rotated') for repeat in range(repeats)]
            for kind, repeat, change in conditions:
                altered = z + change
                logits = model.unembed(altered).double()
                for split in ('train', 'heldout'):
                    idx = indices[split]
                    row = dict(checkpoint=str(Path(path).resolve()), step=saved['step'],
                               strength=strength, condition=kind, repeat=repeat, split=split,
                               loss=F.cross_entropy(logits[idx], y[idx]).item(),
                               accuracy=(logits[idx].argmax(1) == y[idx]).double().mean().item(),
                               displacement_rms=change[idx].square().sum(1).mean().sqrt().item(),
                               **geometry(altered[idx], groups[idx]))
                    rows.append(row)
        print(f"Analyzed step {saved['step']}: {path}", flush=True)
    (output / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (output / 'protocol.json').write_text(json.dumps(dict(
        clusters=clusters, strengths=list(strengths), repeats=repeats, seed=seed,
        representation='post_block_residual, equals token, before unembedding',
        fitting='all training activations only; raw Euclidean k-means',
        checkpoints=[str(Path(p).resolve()) for p in checkpoints],
        torch_version=str(torch.__version__), source=Path(__file__).read_text()), indent=2),
        encoding='utf-8')
    from .intervention_report import write_report
    if 0. in strengths and any(s > 0 for s in strengths):
        write_report(output)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=601)
    parser.add_argument('--strengths', nargs='+', type=float, default=[0., .25, .5, 1.])
    args = parser.parse_args()
    paths = sorted(set(p for pattern in args.checkpoints for p in glob.glob(pattern)))
    if not paths:
        parser.error('no checkpoints matched')
    torch.set_num_threads(1)
    sweep(paths, args.output_dir, args.clusters, args.strengths, args.repeats, args.seed)


if __name__ == '__main__':
    main()
