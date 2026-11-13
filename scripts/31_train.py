"""Phase 3 · Train the classifier of one side of the card with five-fold cross-validation.

It connects to nothing: it reads the manifest, the split (30_make_splits.py) and the photos on
disk, and trains on this machine (GPU when there is one).

For each fold, one model is trained on three folds, watched on a fourth (early stopping) and
used to score the fifth, which it never saw. At the end every person has a score given by a
model that did not train on them (or on their ring), and the metrics are computed on those.

Everything is left in runs/<name>/ (never in git):
  config.json            what was run: arguments, seed, versions, commit
  history.csv            loss and validation PR-AUC of every epoch of every fold
  oof_predictions.csv    anon_id, fold, label, label source, channel and score
  metrics_*.csv          overall, by fold, by channel and by label source
  fold<k>.pt             weights of the model of each fold

With --final it trains the model for new customers instead: one network on everyone, with no
fold held out and a fixed number of epochs taken from a cross-validation run of the same recipe
(--epochs-from: the median of the epochs its five folds kept). That model has no score of its
own to report; its expected result is the one of the cross-validation. It leaves final.pt.

Usage:  python scripts/31_train.py --side back
        python scripts/31_train.py --side front [--arch efficientnet_b0] [--size 512x320]
        python scripts/31_train.py --side back --limit 400 --epochs 2 --folds 0     quick trial
        python scripts/31_train.py --side front --arch efficientnet_b0 --final --epochs-from runs/<cv run>
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from id_fraud_detector.common import ROOT, image_path, load_config, output_dir
from id_fraud_detector.evaluation import metrics
from id_fraud_detector.model.data import build_cache, excluded_by_review
from id_fraud_detector.model.net import ARCHITECTURES, build
from id_fraud_detector.model.train import TrainConfig, fit, predict

MAX_FPR = 0.02  # main metric: recall when at most 2 % of the genuine IDs are flagged


def people_table(out, limit: int | None, seed: int) -> pd.DataFrame:
    splits = out / "splits.csv"
    if not splits.exists():
        sys.exit("ERROR: there is no split yet. Run `python scripts/30_make_splits.py` first.")
    manifest = pd.read_csv(out / "manifest.csv", usecols=["anon_id", "tenant", "label", "label_source", "channel"])
    people = pd.read_csv(splits).merge(manifest, on="anon_id", how="inner", validate="one_to_one")
    people["channel"] = people["channel"].fillna("(none)")
    people["positive"] = people["label"] == "POSITIVE"
    # Kept in the table (the cache of photos holds them) but never trained on, watched or scored.
    people["excluded"] = people["anon_id"].isin(excluded_by_review(out)) & people["positive"]
    if limit is not None and limit < len(people):  # quick trial: keep the share of positives
        positives = people[people["positive"]]
        keep_positives = max(10, round(limit * len(positives) / len(people)))
        people = pd.concat([positives.sample(min(len(positives), keep_positives), random_state=seed),
                            people[~people["positive"]].sample(limit - keep_positives, random_state=seed)])
    return people.sort_values("anon_id").reset_index(drop=True)


def kept_epochs(run) -> tuple[list[int], int]:
    """Epoch kept by each fold of a cross-validation run (counted through both stages) and how
    many of its epochs trained only the head."""
    history = pd.read_csv(run / "history.csv")
    kept = [int(part.reset_index(drop=True)["val_pr_auc"].idxmax()) + 1 for _, part in history.groupby("fold")]
    head = int((history[history["fold"] == history["fold"].iloc[0]]["stage"] == "head").sum())
    return kept, head


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def report(run_dir, oof: pd.DataFrame) -> None:
    overall = metrics.summary(oof["positive"], oof["score"], MAX_FPR)
    threshold = metrics.threshold_at_fpr(oof["positive"], oof["score"], MAX_FPR)
    tables = {
        "overall": pd.DataFrame([overall]),
        "by_fold": pd.DataFrame([{"fold": fold, **metrics.summary(part["positive"], part["score"], MAX_FPR)}
                                 for fold, part in oof.groupby("fold")]),
        "by_channel": metrics.by_group(oof, "channel", threshold),
        "by_label_source": metrics.by_group(oof, "label_source", threshold)
        .drop(columns=["roc_auc_inside_group"]),
    }
    titles = {"overall": "Every person, scored by a model that never saw them",
              "by_fold": "By fold (each fold has its own model)",
              "by_channel": f"By channel, with the single threshold that flags {MAX_FPR:.0%} of all the negatives",
              "by_label_source": "By label source, with the same threshold"}
    for name, table in tables.items():
        table.to_csv(run_dir / f"metrics_{name}.csv", index=False)
        print(f"\n== {titles[name]}")
        print(table.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=["front", "back"])
    ap.add_argument("--arch", default="mobilenet_v3_large", choices=ARCHITECTURES)
    ap.add_argument("--size", default="512x320", help="input size, width x height")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="comparison only: start from random weights instead of ImageNet (no transfer learning). "
                         "It then trains the whole network from the first epoch, longer and with a higher rate")
    ap.add_argument("--head-epochs", type=int, help="epochs training only the new head (default 3; 0 from scratch)")
    ap.add_argument("--epochs", type=int,
                    help="most epochs of the second stage, early stopping decides (default 20; 40 from scratch)")
    ap.add_argument("--blocks", type=int, help="blocks of the base released in the second stage (default 6; all from scratch)")
    ap.add_argument("--lr", type=float, help="learning rate of the second stage (default 1e-4; 1e-3 from scratch)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=6, help="epochs without improving before stopping")
    ap.add_argument("--dropout", type=float, help="dropout before the output (default: the 0.2 of the network)")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay of the optimizer")
    ap.add_argument("--seed", type=int, help="seed of the weights and of the batch order (default: the one in config.yaml). "
                                             "The split does not change")
    ap.add_argument("--folds", help="test folds to run, for example 0,1 (default: all)")
    ap.add_argument("--limit", type=int, help="quick trial with this many people")
    ap.add_argument("--name", help="name of the run folder (default: side, architecture, size and time)")
    ap.add_argument("--final", action="store_true", help="train the model for new customers: everyone, no fold held out")
    ap.add_argument("--epochs-from", metavar="RUN", help="with --final: cross-validation run that gives the number of epochs")
    args = ap.parse_args()

    cfg = load_config()
    out = output_dir(cfg)
    size = tuple(int(v) for v in args.size.lower().split("x"))
    people = people_table(out, args.limit, cfg["seed"])
    n_folds = int(people["fold"].max()) + 1
    test_folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(n_folds))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    used = ~people["excluded"].to_numpy()
    print(f"people: {int(used.sum())}  positives: {int((people['positive'] & used).sum())}  side: {args.side}  "
          f"model: {args.arch}  input: {size[0]}x{size[1]}")
    if not used.all():
        print(f"left out: {int((~used).sum())} positives that the visual review found to be genuine documents")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU (no GPU found: this will be slow)'}")

    paths = [image_path(out, r.tenant, r.label, r.anon_id, args.side) for r in people.itertuples(index=False)]
    photos = build_cache(out, args.side, size, list(people["anon_id"]), paths,
                         tag=f"_limit{args.limit}" if args.limit else "")
    labels = people["positive"].to_numpy().astype(np.int64)
    folds = people["fold"].to_numpy()

    scratch = args.no_pretrained
    seed = cfg["seed"] if args.seed is None else args.seed

    def choice(given, pretrained_default, scratch_default):
        return given if given is not None else (scratch_default if scratch else pretrained_default)

    reference = None
    if args.final:
        if args.folds:
            sys.exit("ERROR: --final trains on everyone; it does not take --folds.")
        if args.epochs_from:
            kept, head = kept_epochs(ROOT / args.epochs_from if not Path(args.epochs_from).is_absolute() else Path(args.epochs_from))
            total = int(np.median(kept))
            args.head_epochs = head if args.head_epochs is None else args.head_epochs
            args.epochs = max(1, total - args.head_epochs)
            reference = {"run": args.epochs_from, "kept_epochs": kept, "median": total}
            print(f"epochs kept by the folds of {args.epochs_from}: {kept} -> {args.head_epochs} of the head + {args.epochs}")
        elif args.epochs is None:
            sys.exit("ERROR: --final needs --epochs-from RUN (or --epochs): there is no validation fold to stop it.")
    elif args.epochs_from:
        sys.exit("ERROR: --epochs-from only goes with --final.")
    name = args.name or (f"{args.side}_{args.arch}{'_scratch' if scratch else ''}_{size[0]}x{size[1]}"
                         f"{'' if args.seed is None else f'_seed{seed}'}{'_final' if args.final else ''}"
                         f"_{datetime.now():%Y%m%d_%H%M}")
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = TrainConfig(head_epochs=choice(args.head_epochs, 3, 0), tune_epochs=choice(args.epochs, 20, 40),
                            trainable_blocks=choice(args.blocks, 6, 99), tune_lr=choice(args.lr, 1e-4, 1e-3),
                            weight_decay=args.weight_decay, batch_size=args.batch, patience=args.patience, seed=seed)
    if scratch:
        print("no transfer learning: random initial weights, the whole network is trained")
    (run_dir / "config.json").write_text(json.dumps({
        "side": args.side, "arch": args.arch, "pretrained": not scratch, "dropout": args.dropout, "size": list(size),
        "people": int(used.sum()), "excluded_by_review": int((~used).sum()),
        "positives": int(labels[used].sum()), "test_folds": [] if args.final else test_folds, "final": args.final,
        "epochs_from": reference, "limit": args.limit, "train": train_cfg.as_dict(),
        "torch": torch.__version__, "device": str(device), "commit": git_commit(),
        "started": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")

    if args.final:
        started = time.time()
        train_idx = np.flatnonzero(used)
        print()
        print(f"== final model: train {len(train_idx)} ({int(labels[train_idx].sum())} positives), nothing held out")
        torch.manual_seed(seed)
        model = build(args.arch, pretrained=not scratch, dropout=args.dropout).to(device)
        history = fit(model, photos, labels, train_idx, None, train_cfg, device)
        torch.save(model.state_dict(), run_dir / "final.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print()
        print(f"final model saved in {run_dir / 'final.pt'}   ({(time.time() - started) / 60:.1f} min)")
        return

    scored, history = [], []
    for fold in test_folds:
        started = time.time()
        val_fold = (fold + 1) % n_folds
        test_idx = np.flatnonzero((folds == fold) & used)
        val_idx = np.flatnonzero((folds == val_fold) & used)
        train_idx = np.flatnonzero((folds != fold) & (folds != val_fold) & used)
        print(f"\n== fold {fold}: train {len(train_idx)} ({int(labels[train_idx].sum())} positives)  "
              f"validation {len(val_idx)} ({int(labels[val_idx].sum())})  test {len(test_idx)} ({int(labels[test_idx].sum())})")
        torch.manual_seed(seed + fold)
        model = build(args.arch, pretrained=not scratch, dropout=args.dropout).to(device)
        fold_cfg = TrainConfig(**{**train_cfg.as_dict(), "seed": seed + fold})
        history += [{"fold": fold, **row} for row in fit(model, photos, labels, train_idx, val_idx, fold_cfg, device)]
        part = people.iloc[test_idx][["anon_id", "fold", "label", "label_source", "channel", "positive"]].copy()
        part["score"] = predict(model, photos, test_idx, device)
        scored.append(part)
        torch.save(model.state_dict(), run_dir / f"fold{fold}.pt")
        pd.concat(scored).to_csv(run_dir / "oof_predictions.csv", index=False)  # saved as it goes
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"   test fold {fold}: {metrics.summary(part['positive'], part['score'], MAX_FPR)}"
              f"   ({(time.time() - started) / 60:.1f} min)")

    oof = pd.concat(scored)
    if len(test_folds) == n_folds:
        report(run_dir, oof)
    else:
        print("\nOnly some folds were run: the overall metrics need all of them.")
    print(f"\nrun saved in {run_dir}")


if __name__ == "__main__":
    main()
