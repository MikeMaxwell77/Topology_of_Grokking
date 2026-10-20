"""Offline H0/H1 measurement of trusted study checkpoints."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from .config import TopologyStudyConfig
from .study import StudyTransformer, build_data, load_checkpoint, rng_state, restore_rng


@contextmanager
def isolated_analysis(seed=600):
    state = rng_state()
    try:
        # Local generator reserved for analysis extensions; persistence itself is deterministic.
        yield np.random.default_rng(seed)
    finally:
        restore_rng(state)


def geometry(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('need at least two finite points')
    centered = points - points.mean(0)
    rms = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if rms == 0:
        return None, rms, None
    normalized = centered / rms
    singular = np.linalg.svd(normalized, compute_uv=False)
    eigenvalues = singular ** 2 / len(points)
    weights = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
    return normalized, rms, effective_rank


def finite_diagram(diagram):
    diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
    if np.isnan(diagram).any() or np.isneginf(diagram).any() or not np.isfinite(diagram[:, 0]).all():
        raise ValueError('invalid persistence diagram')
    if (diagram[:, 1] < diagram[:, 0]).any():
        raise ValueError('death must be >= birth')
    return diagram[np.isfinite(diagram[:, 1])]


def wasserstein(first, second):
    """persim 1-Wasserstein, Euclidean ground cost, optimal diagonal matching."""
    first, second = finite_diagram(first), finite_diagram(second)
    if not len(first):
        return float(((second[:, 1] - second[:, 0]) / np.sqrt(2)).sum())
    if not len(second):
        return float(((first[:, 1] - first[:, 0]) / np.sqrt(2)).sum())
    from persim import wasserstein as distance
    return float(distance(first, second))


def betti_curve(diagram, scales):
    diagram = np.asarray(diagram).reshape(-1, 2)
    return ((diagram[:, :1] <= scales) & (scales < diagram[:, 1:])).sum(0)


def measure(points, scales):
    from ripser import ripser
    if not np.isfinite(points).all():
        raise ValueError('nonfinite representation')
    diagrams = ripser(np.asarray(points), maxdim=1)['dgms']
    metrics, curves = {}, []
    for dim, diagram in enumerate(diagrams):
        finite = finite_diagram(diagram)
        persistence = finite[:, 1] - finite[:, 0]
        metrics.update({f'h{dim}_finite_feature_count': len(finite),
                        f'h{dim}_essential_feature_count': len(diagram) - len(finite),
                        f'h{dim}_total_finite_persistence': float(persistence.sum()),
                        f'h{dim}_mean_finite_persistence': float(persistence.mean()) if len(finite) else 0.,
                        f'h{dim}_max_finite_persistence': float(persistence.max()) if len(finite) else 0.})
        curves.append(betti_curve(diagram, scales))
    return metrics, diagrams, curves


def qualifying_start(history, key, threshold, consecutive):
    for i in range(len(history) - consecutive + 1):
        if all(row[key] >= threshold for row in history[i:i + consecutive]):
            return history[i]['step']
    return None


def behavioral_timing(history, train_threshold=.99, heldout_threshold=.95, consecutive=5,
                      poor_threshold=.5, plateau_min_steps=500):
    memorization = qualifying_start(history, 'train_accuracy', train_threshold, consecutive)
    generalization = qualifying_start(history, 'heldout_accuracy', heldout_threshold, consecutive)
    # Poor plateau: a contiguous measured sequence after memorization, before generalization.
    longest, begin, last = 0, None, None
    for row in history:
        eligible = (memorization is not None and row['step'] >= memorization and
                    (generalization is None or row['step'] < generalization) and
                    row['train_accuracy'] >= train_threshold and row['heldout_accuracy'] < poor_threshold)
        if eligible:
            begin = row['step'] if begin is None else begin
            longest = max(longest, row['step'] - begin)
            last = row['step']
        else:
            begin, last = None, None
    return {'memorization_step': memorization, 'generalization_step': generalization,
            'delay_steps': generalization - memorization if memorization is not None and generalization is not None else None,
            'poor_plateau_max_measured_span': longest,
            'poor_plateau_detected': longest >= plateau_min_steps,
            'train_threshold': train_threshold, 'heldout_threshold': heldout_threshold,
            'consecutive_evaluations': consecutive, 'poor_threshold': poor_threshold,
            'plateau_min_steps': plateau_min_steps,
            'evaluation_spacings': sorted(set(np.diff([r['step'] for r in history]).tolist())),
            'absence_definition': 'null event means absent within observed evaluation budget'}


def export_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(checkpoints, output_dir, *, scales=None, analysis_seed=None, timing_options=None):
    with isolated_analysis(analysis_seed):
        return _analyze(checkpoints, output_dir, scales=scales,
                        analysis_seed=analysis_seed, timing_options=timing_options)


def _analyze(checkpoints, output_dir, *, scales=None, analysis_seed=None, timing_options=None):
    checkpoints = sorted((Path(p) for p in checkpoints), key=lambda p: int(p.stem.split('_')[-1]))
    if not checkpoints:
        raise ValueError('no checkpoints selected')
    if analysis_seed is None:
        analysis_seed = load_checkpoint(checkpoints[0])['config']['analysis_seed']
    scales = np.linspace(0., 4., 201) if scales is None else np.asarray(scales)
    if scales.ndim != 1 or len(scales) < 2 or not np.isfinite(scales).all() or (np.diff(scales) <= 0).any():
        raise ValueError('filtration scales must be finite and strictly increasing')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('analysis requires a fresh output directory to preserve previous measurements')
    rows, previous, history = [], {}, []
    reference = None
    with isolated_analysis(analysis_seed):
        for path in checkpoints:
            saved = load_checkpoint(path)
            config = TopologyStudyConfig(**saved['config'])
            identity = (saved['config'], saved['source'])
            if reference is None:
                reference = identity
                reference_indices = saved['indices']
            if identity != reference or any(not torch.equal(saved['indices'][k], reference_indices[k])
                                            for k in ('train', 'heldout', 'probes')):
                raise ValueError('checkpoints must belong to the same run with identical splits and probes')
            history = saved['history']
            model = StudyTransformer(config)
            model.load_state_dict(saved['model'])
            model.eval()
            inputs, labels, _ = build_data(config)
            probes = saved['indices']['probes']
            with torch.no_grad():
                clouds = {key: value.numpy() for key, value in model.representations(inputs[probes]).items()}
            step = saved['step']
            arrays = dict(inputs=inputs[probes].numpy(), pairs=inputs[probes, :2].numpy(),
                          labels=labels[probes].numpy(), probe_indices=probes.numpy(), step=step,
                          token_position=2, layer='block.0', **clouds)
            np.savez_compressed(output / f'representations_{step:08d}.npz', **arrays)
            for representation, raw in clouds.items():
                try:
                    normalized, rms, rank = geometry(raw)
                    geometry_error = None
                except (ValueError, np.linalg.LinAlgError) as error:
                    normalized, rms, rank, geometry_error = None, None, None, str(error)
                for mode, cloud in [('raw', raw), ('rms_normalized', normalized)]:
                    key = (representation, mode)
                    row = dict(step=step, representation=representation, geometry=mode,
                               representation_rms=rms, covariance_effective_rank=rank,
                               valid=False, error=None, previous_step=None, spacing_steps=None)
                    try:
                        if geometry_error:
                            raise ValueError(geometry_error)
                        if cloud is None:
                            raise ValueError('collapsed representation: RMS normalization undefined')
                        metrics, diagrams, curves = measure(cloud, scales)
                        row.update(metrics, valid=True)
                        if key in previous:
                            old_step, old_diagrams = previous[key]
                            spacing = step - old_step
                            if spacing <= 0:
                                raise ValueError('duplicate or unordered checkpoint steps')
                            row.update(previous_step=old_step, spacing_steps=spacing)
                            for dim in range(2):
                                shift = wasserstein(old_diagrams[dim], diagrams[dim])
                                row[f'h{dim}_wasserstein_shift'] = shift
                                row[f'h{dim}_wasserstein_shift_per_step'] = shift / spacing
                        previous[key] = (step, diagrams)
                        np.savez_compressed(output / f'topology_{representation}_{mode}_{step:08d}.npz',
                                            h0=diagrams[0], h1=diagrams[1], scales=scales,
                                            betti_h0=curves[0], betti_h1=curves[1])
                    except Exception as error:
                        row.update(valid=False, error=f'{type(error).__name__}: {error}')
                        previous.pop(key, None)  # never bridge across a failed measurement
                    rows.append(row)
    timing = behavioral_timing(history, **(timing_options or {}))
    sensitivity = [behavioral_timing(history, t, h, n) for t in (.98, .99)
                   for h in (.90, .95, .99) for n in (3, 5)]
    metadata = {'primary_outcome': 'post_block_residual rms_normalized h1_total_finite_persistence',
                'normalization': '(X - centroid) / sqrt(mean(sum((X-centroid)^2, axis=1)))',
                'effective_rank': 'exp(-sum(p_i log p_i)), p_i = covariance eigenvalue / trace; covariance=X_centered.T X_centered/N',
                'wasserstein': 'persim: 1-Wasserstein with Euclidean ground metric and diagonal matching; essential intervals excluded',
                'filtration_scales': scales.tolist(), 'analysis_seed': analysis_seed,
                'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'ripser', 'persim', 'matplotlib')},
                'checkpoints': [str(p.resolve()) for p in checkpoints], 'config': asdict(config),
                'analysis_source': Path(__file__).read_text(encoding='utf-8'),
                'timing': timing, 'timing_sensitivity': sensitivity}
    (output / 'analysis.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (output / 'metrics.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    export_csv(output / 'metrics.csv', rows)
    export_csv(output / 'training.csv', history)
    report = [f'Measured {len(checkpoints)} checkpoints; {sum(not r["valid"] for r in rows)} failed geometry/topology measurements.',
              f'Memorization: {timing["memorization_step"]}; generalization: {timing["generalization_step"]}; delay: {timing["delay_steps"]} steps.',
              f'Poor-generalization plateau detected: {timing["poor_plateau_detected"]} (measured span {timing["poor_plateau_max_measured_span"]} steps).',
              'A null event is absent within the observed budget. Timing uses behavior alone.',
              'These are observations. Aligned curves establish neither causation nor a circuit.',
              'Pilot acceptance requires inspecting the behavioral curves and timing sensitivity before fresh seeds or a sweep.']
    (output / 'report.txt').write_text('\n'.join(report))
    plot_outputs(output, rows, history)
    return rows, metadata


def plot_outputs(output, rows, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(9, 1, figsize=(11, 22), sharex=True)
    for name in ('train', 'heldout'):
        axes[0].plot([r['step'] for r in history], [r[f'{name}_accuracy'] for r in history], label=name)
        axes[1].plot([r['step'] for r in history], [r[f'{name}_loss'] for r in history], label=name)
    axes[6].plot([r['step'] for r in history], [r['parameter_l2'] for r in history], label='parameter L2')
    for representation in ('post_block_residual', 'mlp_activation'):
        for mode in ('raw', 'rms_normalized'):
            selected = [r for r in rows if r['representation'] == representation and r['geometry'] == mode]
            x = [r['step'] for r in selected]
            offset = 0 if mode == 'raw' else 1
            axes[2 + offset].plot(x, [r.get('h1_total_finite_persistence', np.nan) if r['valid'] else np.nan for r in selected], label=representation)
            axes[4 + offset].plot(x, [r.get('h1_wasserstein_shift_per_step', np.nan) if r['valid'] else np.nan for r in selected], label=representation)
        selected = [r for r in rows if r['representation'] == representation and r['geometry'] == 'raw']
        axes[7].plot([r['step'] for r in selected], [r['representation_rms'] for r in selected], label=representation)
        axes[8].plot([r['step'] for r in selected], [r['covariance_effective_rank'] for r in selected], label=representation)
    for axis, title in zip(axes, ['Accuracy', 'Cross-entropy', 'Raw H1 persistence', 'Normalized H1 persistence',
                                 'Raw H1 Wasserstein / step', 'Normalized H1 Wasserstein / step',
                                 'Parameter L2', 'Representation RMS', 'Covariance effective rank']):
        axis.set_ylabel(title, fontsize=9)
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel('Completed optimizer steps')
    fig.tight_layout()
    fig.savefig(output / 'aligned_metrics.png')
    plt.close(fig)
    # All measured checkpoints are available, with common axes within each geometry.
    for mode in ('raw', 'rms_normalized'):
        files = sorted(output.glob(f'topology_*_{mode}_*.npz'))
        if not files:
            continue
        diagrams = [np.load(p) for p in files]
        maximum = max([float(finite_diagram(d[f'h{k}']).max()) for d in diagrams for k in (0, 1)
                       if len(finite_diagram(d[f'h{k}']))] + [1.]) * 1.05
        max_betti = max(int(d[f'betti_h{k}'].max()) for d in diagrams for k in (0, 1))
        for path, data in zip(files, diagrams):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for dim in (0, 1):
                finite = finite_diagram(data[f'h{dim}'])
                axes[0].scatter(finite[:, 0], finite[:, 1], label=f'H{dim}')
                axes[1].plot(data['scales'], data[f'betti_h{dim}'], label=f'H{dim}')
            axes[0].plot([0, maximum], [0, maximum], color='gray')
            axes[0].set(xlim=(0, maximum), ylim=(0, maximum), xlabel='Birth', ylabel='Death (finite intervals)')
            axes[1].set(xlim=(data['scales'][0], data['scales'][-1]), ylim=(0, max_betti + 1), xlabel='Filtration scale', ylabel='Betti number (includes essential H0)')
            for axis in axes:
                axis.legend()
            fig.suptitle(path.stem)
            fig.tight_layout()
            fig.savefig(output / (path.stem + '.png'))
            plt.close(fig)
            data.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True, help='paths or glob patterns')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scale-max', type=float, default=4.)
    parser.add_argument('--scale-count', type=int, default=201)
    parser.add_argument('--analysis-seed', type=int)
    parser.add_argument('--train-threshold', type=float, default=.99)
    parser.add_argument('--heldout-threshold', type=float, default=.95)
    parser.add_argument('--consecutive', type=int, default=5)
    parser.add_argument('--poor-threshold', type=float, default=.5)
    parser.add_argument('--plateau-min-steps', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    import glob
    paths = sorted(set(p for pattern in args.checkpoints for p in glob.glob(pattern)))
    torch.set_num_threads(args.threads)
    if args.consecutive < 1 or args.plateau_min_steps < 0 or any(not 0 <= t <= 1 for t in
            (args.train_threshold, args.heldout_threshold, args.poor_threshold)):
        parser.error('invalid timing settings')
    analyze(paths, args.output_dir, scales=np.linspace(0, args.scale_max, args.scale_count),
            analysis_seed=args.analysis_seed, timing_options={k: getattr(args, k) for k in
            ('train_threshold', 'heldout_threshold', 'consecutive', 'poor_threshold', 'plateau_min_steps')})


if __name__ == '__main__':
    main()
