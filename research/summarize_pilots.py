"""Export compact results and SVG figures from trusted local development runs."""
import json
import shutil
from pathlib import Path
from grokking.circuits import behavior_timing

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'research'
RUNS = ['circuit_pilot', 'circuit_p31_validation', 'circuit_scaled_p31', 'circuit_baseline_p113']


def summarize_analysis(source):
    rows = json.loads(source.read_text())['rows']
    summary = []
    for row in rows:
        interventions = []
        for item in row['interventions']:
            selected = {k: item[k] for k in ['name', 'heldout']}
            if not item['name'].startswith('heads'):
                selected.update({k: item[k] for k in ['rank', 'removed_norm',
                    'prediction_change_rate', 'topology_distance_from_clean']})
            interventions.append(selected)
        summary.append({
            'step': row['step'], 'train': row['train'], 'heldout': row['heldout'],
            'candidate_frequencies': row.get('candidate_frequencies'),
            'mlp_equals_fourier': {k: v for k, v in row['features']['mlp.equals'].items() if k != 'power'},
            'topology': {k: {
                'H1_total_persistence': v['dimensions'][1]['total_persistence'],
                'H1_max_persistence': v['dimensions'][1]['max_persistence'],
                'rms_scale': v['rms_scale'], 'n': v['n'],
                'H1_half_probe_total_persistence': v['nested_half'][1]['total_persistence'],
                'H1_wasserstein_previous': v['wasserstein_previous'][1] if v['wasserstein_previous'] else None,
            } for k, v in row['topology'].items()},
            'interventions': interventions,
        })
    return summary


def main():
    behavior = []
    for name in RUNS:
        root = ROOT / 'outputs' / name
        if not (root / 'timing.json').exists():
            continue  # Never label an in-progress run complete.
        rows = json.loads((root / 'behavior.json').read_text())
        behavior.append({'run': name, 'config': json.loads((root / 'config.json').read_text()),
                         'initial': rows[0], 'final': rows[-1], 'timing': behavior_timing(rows)})
        source = root / 'analysis_fixed' / 'measurements.json'
        if source.exists():
            summary = summarize_analysis(source)
            (DEST / f'{name}_summary.json').write_text(json.dumps(summary, indent=2))
            for figure in ['aligned.svg', 'interventions.svg']:
                shutil.copyfile(source.parent / figure, DEST / f'{name}_{figure}')
    (DEST / 'behavior_summary.json').write_text(json.dumps(behavior, indent=2))


if __name__ == '__main__':
    main()
