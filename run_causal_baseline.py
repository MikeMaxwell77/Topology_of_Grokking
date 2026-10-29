"""Run the untouched CPU pilot, then the exploratory evaluation-only sweep."""
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    subprocess.run([sys.executable, '-u', '-m', 'grokking.study', 'train',
                    '--device', 'cpu', '--threads', '1', '--output-dir',
                    'outputs/cluster_causal_baseline'], cwd=root, check=True)
    subprocess.run([sys.executable, '-u', '-m', 'grokking.interventions',
                    '--checkpoints', 'outputs/cluster_causal_baseline/checkpoint_*000.pt',
                    '--output-dir', 'outputs/cluster_causal_sweep'], cwd=root, check=True)
