"""Experiment 4: evaluation-only inversion of saved activation perturbations."""
import argparse
import glob
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import TopologyStudyConfig
from .interventions import control_delta, displacement, fit_clusters, geometry
from .study import StudyTransformer, build_data, load_checkpoint, provenance


@torch.no_grad()
def sweep(checkpoints, output_dir, clusters=8, strengths=(0., .25, .5, 1.),
          repeats=5, seed=601):
    paths = list(checkpoints)
    strengths = list(strengths)
    if not paths or not strengths or repeats < 1 or any(not 0 <= s <= 1 for s in strengths):
        raise ValueError('checkpoints, strengths in [0,1], and positive repeats required')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(checkpoints=[str(Path(p).resolve()) for p in paths],
                    clusters=clusters, strengths=strengths, repeats=repeats, seed=seed,
                    representation='equals-token post-block residual before unembedding',
                    rescue='subtract the identical saved displacement; no refitting',
                    fitting='training-only k-means; held-out nearest-center assignment',
                    torch_version=str(torch.__version__), source=provenance(),
                    runner_source=Path(__file__).read_text())
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows = []
    for number, path in enumerate(paths):
        saved = load_checkpoint(path)
        config = TopologyStudyConfig(**saved['config'])
        model = StudyTransformer(config).eval()
        model.load_state_dict(saved['model'])
        x, y, _ = build_data(config)
        indices = saved['indices']
        z = torch.cat([model.representations(b)['post_block_residual']
                       for b in x.split(config.evaluation_batch_size)])
        train = z[indices['train']]
        centers = fit_clusters(train, clusters, seed)
        mean = train.mean(0)
        clean_logits = model.unembed(z).double()
        perturbations = []
        for strength in strengths:
            delta, groups = displacement(z, centers, mean, strength)
            conditions = [('targeted', -1, delta)]
            if strength:
                conditions += [(kind, r, control_delta(delta, kind, seed + 1 + r))
                               for kind in ('random', 'rotated') for r in range(repeats)]
            for kind, repeat, change in conditions:
                perturbations.append(dict(strength=strength, condition=kind,
                                          repeat=repeat, delta=change))
                disrupted = z + change
                restored = disrupted - change
                for mode, residual in [('clean', z), ('disrupted', disrupted),
                                       ('rescued', restored)]:
                    logits = model.unembed(residual).double()
                    for split in ('train', 'heldout'):
                        idx = indices[split]
                        loss = F.cross_entropy(logits[idx], y[idx]).item()
                        clean_loss = F.cross_entropy(clean_logits[idx], y[idx]).item()
                        rows.append(dict(checkpoint=str(Path(path).resolve()), step=saved['step'],
                                         strength=strength, condition=kind, repeat=repeat,
                                         mode=mode, split=split, loss=loss,
                                         loss_difference_from_clean=loss - clean_loss,
                                         accuracy=(logits[idx].argmax(1) == y[idx]).double().mean().item(),
                                         max_activation_error=(residual[idx] - z[idx]).abs().max().item(),
                                         max_logit_error=(logits[idx] - clean_logits[idx]).abs().max().item(),
                                         **geometry(residual[idx], groups[idx])))
        torch.save(dict(checkpoint=str(Path(path).resolve()), centers=centers, mean=mean,
                        groups=groups, indices=indices, perturbations=perturbations),
                   output / f'perturbations_{number:08d}.pt')
        (output / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f"Rescue evaluated at step {saved['step']}: {path}", flush=True)
    rescued = [r for r in rows if r['mode'] == 'rescued']
    report = (
        '# Experiment 4: inverse-perturbation rescue\n\n'
        'This is an algebraic implementation control, not independent causal evidence.\n\n'
        f"Maximum rescued activation error: {max(r['max_activation_error'] for r in rescued):.8g}\n\n"
        f"Maximum rescued logit error: {max(r['max_logit_error'] for r in rescued):.8g}\n\n"
        f"Maximum absolute rescued loss difference: {max(abs(r['loss_difference_from_clean']) for r in rescued):.8g}\n\n"
        'Inspect all splits, strengths, conditions and checkpoints in results.json. '
        'Floating-point roundoff can prevent exact recovery.\n')
    (output / 'report.md').write_text(report, encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--strengths', nargs='+', type=float, default=[0., .25, .5, 1.])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=601)
    args = vars(parser.parse_args())
    paths = sorted(set(p for pattern in args.pop('checkpoints') for p in glob.glob(pattern)))
    if not paths:
        parser.error('no checkpoints matched')
    torch.set_num_threads(1)
    sweep(paths, **args)


if __name__ == '__main__':
    main()
