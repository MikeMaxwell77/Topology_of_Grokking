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
    parser.add_argument("--start-epoch", type=int, help="first epoch to include in plots")
    parser.add_argument("--end-epoch", type=int, help="last epoch to include in plots")
    return parser.parse_args()


def restrict_epochs(history: dict, start_epoch: int | None, end_epoch: int | None) -> dict:
    """Return a history containing only checkpoints in the requested epoch range."""
    if start_epoch is None and end_epoch is None:
        return history
    epochs = history.get("epoch", [])
    selected = [
        index for index, epoch in enumerate(epochs)
        if (start_epoch is None or epoch >= start_epoch)
        and (end_epoch is None or epoch <= end_epoch)
    ]
    restricted = dict(history)
    for key, value in history.items():
        if isinstance(value, list) and len(value) == len(epochs):
            restricted[key] = [value[index] for index in selected]
    return restricted


def main() -> None:
    arguments = parse_args()
    history = load_history(arguments.history)
    history = restrict_epochs(history, arguments.start_epoch, arguments.end_epoch)
    print(format_summary(summarize_history(history, grokking_threshold=arguments.threshold)))
    if arguments.plot:
        paths = save_tracking_plots(history, arguments.plot_dir)
        print(f"Saved {len(paths)} graphs in {arguments.plot_dir}.")


if __name__ == "__main__":
    main()
