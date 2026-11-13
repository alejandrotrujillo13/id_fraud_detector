"""Phase 3 · Training curves of a run: training loss against validation loss, by fold.

It connects to nothing. It reads runs/<name>/history.csv (written by 31_train.py) and draws one
panel per fold: the loss on the training photos and on the validation fold after each epoch,
with a mark on the epoch whose weights were kept (the best validation PR-AUC). When the
validation loss turns up while the training loss keeps going down, the network is overfitting;
the mark shows where early stopping left it.

The figure goes into the report, which is in Spanish, so its texts are in Spanish.

Usage:  python scripts/34_plot_history.py runs/<name> [--metric loss|accuracy]
Output: runs/<name>/curves_<metric>.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
TRAIN, VALIDATION = "#2a78d6", "#eb6834"
TEXTS = {"loss": ("Pérdida", "Pérdida de entrenamiento y de validación"),
         "accuracy": ("Exactitud (umbral 0,5)", "Exactitud de entrenamiento y de validación")}
SIDES = {"front": "frente", "back": "reverso"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--metric", default="loss", choices=sorted(TEXTS))
    args = ap.parse_args()

    run = Path(args.run)
    history = pd.read_csv(run / "history.csv")
    column = f"val_{args.metric}"
    if column not in history:
        raise SystemExit(f"ERROR: {run / 'history.csv'} has no {column}: the run is older than that measure. "
                         "Train it again with 31_train.py.")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    folds = sorted(history["fold"].unique())
    axis_label, title = TEXTS[args.metric]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    figure, axes = plt.subplots(1, len(folds), figsize=(3.1 * len(folds), 3.3), sharey=True, facecolor=SURFACE,
                                squeeze=False)
    for axis, fold in zip(axes[0], folds):
        part = history[history["fold"] == fold].reset_index(drop=True)
        x = part.index + 1  # epochs counted through both stages
        kept = int(part["val_pr_auc"].idxmax())
        head_epochs = int((part["stage"] == "head").sum())
        axis.set_facecolor(SURFACE)
        axis.grid(axis="y", color=GRID, linewidth=1)
        axis.set_axisbelow(True)
        if 0 < head_epochs < len(part):  # where the backbone is unfrozen
            axis.axvline(head_epochs + 0.5, color=GRID, linewidth=1)
        axis.axvline(kept + 1, color=MUTED, linewidth=1)
        axis.plot(x, part[f"train_{args.metric}"], color=TRAIN, linewidth=2, solid_capstyle="round", label="Entrenamiento")
        axis.plot(x, part[column], color=VALIDATION, linewidth=2, solid_capstyle="round", label="Validación")
        axis.plot([kept + 1], [part[column][kept]], "o", color=VALIDATION, markersize=8, markeredgecolor=SURFACE,
                  markeredgewidth=2)
        axis.set_title(f"Parte {fold}: se conserva la época {kept + 1} de {len(part)}", fontsize=9, color=INK, loc="left")
        axis.set_xlabel("Época")
        axis.set_xlim(0.5, history.groupby("fold").size().max() + 0.5)
        axis.xaxis.set_major_locator(MultipleLocator(5))
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(length=0)
    axes[0][0].set_ylabel(axis_label)
    axes[0][0].set_ylim(bottom=0)
    if args.metric == "accuracy":
        axes[0][0].set_ylim(0, 1)

    train = config["train"]
    figure.suptitle(f"{title} · {config['arch']}, {SIDES[config['side']]}", x=0.01, ha="left", fontsize=11, color=INK)
    figure.text(0.01, 0.9, f"Línea vertical gris: época que se conserva (mejor PR-AUC de validación). "
                f"Las primeras {train['head_epochs']} épocas entrenan solo la capa final. Paciencia: {train['patience']}.",
                fontsize=8, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.995, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    target = run / f"curves_{args.metric}.png"
    figure.savefig(target, dpi=160, facecolor=SURFACE)
    print(f"saved {target}")


if __name__ == "__main__":
    main()
