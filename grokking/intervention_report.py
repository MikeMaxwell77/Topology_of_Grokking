"""Paired loss contrasts for an evaluation-only intervention sweep (no SciPy)."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def contrasts(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['checkpoint'], row['split']), []).append(row)
    result = []
    for (checkpoint, split), group in groups.items():
        baseline = next(r for r in group if r['strength'] == 0)
        for target in group:
            if target['condition'] != 'targeted' or target['strength'] == 0:
                continue
            for kind in ('random', 'rotated'):
                controls = [r for r in group if r['strength'] == target['strength']
                            and r['condition'] == kind]
                if not controls:
                    raise ValueError(f'missing {kind} controls at {checkpoint}')
                losses = [r['loss'] for r in controls]
                result.append(dict(
                    checkpoint=checkpoint, step=target['step'], split=split,
                    strength=target['strength'], control=kind, draws=len(controls),
                    baseline_loss=baseline['loss'], baseline_accuracy=baseline['accuracy'],
                    targeted_loss_increase=target['loss'] - baseline['loss'],
                    control_mean_loss_increase=mean(losses) - baseline['loss'],
                    excess_targeted_loss=target['loss'] - mean(losses),
                    control_loss_sd=stdev(losses) if len(losses) > 1 else None,
                    targeted_accuracy=target['accuracy'],
                    control_mean_accuracy=mean(r['accuracy'] for r in controls),
                    baseline_between_fraction=baseline['between_fraction'],
                    targeted_between_fraction=target['between_fraction'],
                    control_mean_between_fraction=mean(r['between_fraction'] for r in controls)
                        if all(r['between_fraction'] is not None for r in controls) else None,
                    max_displacement_rms_mismatch=max(abs(r['displacement_rms'] -
                        target['displacement_rms']) for r in controls)))
    return result


def write_report(output):
    output = Path(output)
    rows = json.loads((output / 'results.json').read_text(encoding='utf-8'))
    summary = contrasts(rows)
    if not summary:
        raise ValueError('need a zero-strength baseline and nonzero interventions')
    with (output / 'contrasts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    heldout = [r for r in summary if r['split'] == 'heldout']
    lines = ['# Experiment 1: exploratory necessity sweep', '',
             'Positive excess loss means the targeted intervention is more damaging than',
             'the mean matched control. This is not a significance test or proof of clustering',
             'causing grokking. Control draws are not independent training seeds.', '',
             'Inspect contrasts.csv for both splits, geometry changes, and norm matching.', '',
             '| Step | Strength | Control | Clean accuracy | Target accuracy | Excess loss | Control loss SD |',
             '|---:|---:|---|---:|---:|---:|---:|']
    for r in heldout:
        sd = 'NA' if r['control_loss_sd'] is None else f"{r['control_loss_sd']:.4f}"
        lines.append(f"| {r['step']} | {r['strength']} | {r['control']} | "
                     f"{r['baseline_accuracy']:.4f} | {r['targeted_accuracy']:.4f} | "
                     f"{r['excess_targeted_loss']:.4f} | {sd} |")
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    write_report(parser.parse_args().output_dir)
