"""Command-line summary for a saved grokking experiment."""

import argparse
from pathlib import Path

from grokking.analysis import format_summary, load_history, summarize_history
from grokking.plotting import plot_history_dashboard, save_tracking_plots

plot_interactive = plot_history_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", nargs="?", default="grokking_history.pkl")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--plot", action="store_true", help="save the topology tracking graphs")
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    history = load_history(arguments.history)
    print(format_summary(summarize_history(history, grokking_threshold=arguments.threshold)))
    if arguments.plot:
        paths = save_tracking_plots(history, arguments.plot_dir)
        print(f"Saved {len(paths)} graphs in {arguments.plot_dir}.")


if __name__ == "__main__":
    main()
