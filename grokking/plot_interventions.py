"""Plot one saved intervention run. Run python -m grokking.plot_interventions --help."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sweep(run, split):
    with (run / "contrasts.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == split]
    strengths = sorted({float(r["strength"]) for r in rows})
    fig, axes = plt.subplots(3, len(strengths), squeeze=False, sharex=True,
                             sharey="row", figsize=(4 * len(strengths), 8))
    for strength, column in zip(strengths, axes.T):
        column[0].set_title(f"Strength {strength:g}")
        for kind in ("random", "rotated"):
            group = sorted((r for r in rows if float(r["strength"]) == strength
                            and r["control"] == kind), key=lambda r: int(r["step"]))
            steps = [int(r["step"]) for r in group]
            column[2].plot(steps, [float(r["excess_targeted_loss"]) for r in group], label=kind)
            if kind == "random":
                for ax, metric in zip(column, ("between_fraction", "loss_increase")):
                    ax.plot(steps, [float(r["targeted_" + metric]) for r in group], label="targeted")
                column[0].plot(steps, [float(r["baseline_between_fraction"]) for r in group],
                               label="clean", color="gray")
        column[1].axhline(0, color="gray", linewidth=.7)
        column[2].axhline(0, color="gray", linewidth=.7)
    axes[0, 0].set_ylabel("Between-cluster variance fraction")
    axes[1, 0].set_ylabel("Targeted loss − clean loss")
    axes[2, 0].set_ylabel("Targeted loss − mean control loss")
    for ax in (axes[0, 0], axes[2, 0]):
        ax.legend()
    return fig


def timing(run, split):
    protocol = json.loads((run / "protocol.json").read_text())
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    colors = dict(clean="black", targeted="tab:red", random="tab:blue", rotated="tab:green")
    for path in sorted(run.glob("*/training.json")):
        family = path.parent.name.split("_")[0]
        history = json.loads(path.read_text())
        for ax, mode in zip(axes, ("clean", "active")):
            rows = sorted((r for r in history if r["split"] == split and r["mode"] == mode),
                          key=lambda r: r["step"])
            control = family in ("random", "rotated")
            ax.plot([r["step"] for r in rows], [r["accuracy"] for r in rows],
                    color=colors[family], alpha=.45 if control else 1,
                    zorder=2 if control else 3,
                    label=family if family not in ax.get_legend_handles_labels()[1] else None)
            ax.set_ylabel(f"{mode.capitalize()} evaluation accuracy")
    for ax in axes:
        ax.axvspan(protocol["start"], protocol["removal_step"], alpha=.12, color="gray",
                   label="intervention window")
        ax.axhline(.95, color="gray", linestyle=":", linewidth=1)
    axes[0].legend(ncol=3)
    return fig


def rescue(run, split):
    rows = json.loads((run / "results.json").read_text())
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for ax, mode in zip(axes, ("disrupted", "rescued")):
        for kind in ("targeted", "random", "rotated"):
            maxima = {}
            for r in rows:
                if (r["split"], r["mode"], r["condition"]) == (split, mode, kind):
                    step = r["step"]
                    maxima[step] = max(maxima.get(step, 0), abs(r["loss_difference_from_clean"]))
            steps = sorted(maxima)
            ax.plot(steps, [maxima[s] for s in steps], label=kind)
        ax.set_title(f"{mode.capitalize()} · maximum across strengths and draws")
        ax.set_ylabel("|Loss − clean loss|")
        ax.ticklabel_format(axis="y", useOffset=False)
    axes[0].legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    plots = {"sweep": sweep, "timing": timing, "rescue": rescue}
    parser.add_argument("kind", choices=plots)
    parser.add_argument("run", type=Path, help="directory containing the experiment report or branch arms")
    parser.add_argument("--split", choices=("train", "heldout"), default="heldout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", nargs=2, type=int, metavar=("START", "END"),
                        help="zoom the training-step axis")
    args = parser.parse_args()
    fig = plots[args.kind](args.run, args.split)
    fig.suptitle(f"{args.run.parent.name} / {args.run.name} · {args.split}")
    if args.steps:
        for ax in fig.axes:
            ax.set_xlim(*args.steps)
    fig.supxlabel("Training step")
    fig.tight_layout()
    path = args.output or args.run / f"{args.kind}_{args.split}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()

