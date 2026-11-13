"""Phase 3 · One score per person from the front and the back models, plus the rules.

It connects to nothing. It reads the scores of runs of 31_train.py (one or more per side; each
score was given by a model that never saw that person) and, when it exists, the result of the
rules (rules_per_customer.csv), and reports:

  - each side alone and three simple ways of combining them (mean, highest, lowest). No
    combination is trained here, so there is nothing to overfit: the mean is the one kept;
  - how much the two sides overlap: caught by both, by one only, by none;
  - the combined score by fold, channel and label source;
  - classifier and rules together: a person is flagged when the score is over the threshold
    or any rule failed.

Everything is saved in runs/<name>/ (oof_predictions.csv has front_score, back_score, the
combined score and whether a rule failed). missed_positives.csv lists the positives the
combined score does not flag: it is the queue for the visual review (13_visual_review.py).

With several runs for a side, the score of that side is the plain mean of their scores (an
ensemble: nothing is fitted, every run weighs the same). All the runs must come from the same
split, so that no model scores a person it saw in training.

Usage:  python scripts/33_combine_sides.py --front runs/<front run> [more] --back runs/<back run> [more]
                                           [--name combined]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from id_fraud_detector.common import ROOT, load_config, output_dir
from id_fraud_detector.evaluation import metrics

MAX_FPR = 0.02


def scores_of(runs: list[str], side: str) -> pd.DataFrame:
    """Scores of one side: those of the run, or the mean over the runs when there are several."""
    first = None
    for i, run in enumerate(map(Path, runs)):
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config["side"] != side:
            raise SystemExit(f"ERROR: {run} is a run of the {config['side']}, not of the {side}.")
        part = pd.read_csv(run / "oof_predictions.csv")
        if first is None:
            first = part.rename(columns={"score": "score_0"})
            continue
        first = first.merge(part[["anon_id", "fold", "score"]].rename(columns={"score": f"score_{i}"}),
                            on=["anon_id", "fold"], how="inner", validate="one_to_one")
        if len(first) != len(part):
            raise SystemExit(f"ERROR: {run} does not hold the same people in the same folds as {runs[0]}.")
    columns = [f"score_{i}" for i in range(len(runs))]
    first["score"] = first[columns].mean(axis=1)
    return first.drop(columns=columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", nargs="+", required=True, metavar="RUN", help="run(s) of the front")
    ap.add_argument("--back", nargs="+", required=True, metavar="RUN", help="run(s) of the back")
    ap.add_argument("--name", default="combined", help="name of the folder under runs/ for the result")
    args = ap.parse_args()

    front, back = scores_of(args.front, "front"), scores_of(args.back, "back")
    df = front.rename(columns={"score": "front_score"}).merge(
        back[["anon_id", "fold", "score"]].rename(columns={"score": "back_score"}), on=["anon_id", "fold"],
        how="inner", validate="one_to_one")
    if len(df) != len(front) or len(df) != len(back):
        raise SystemExit("ERROR: the two runs do not hold the same people in the same folds.")
    df["score"] = df[["front_score", "back_score"]].mean(axis=1)
    y = df["positive"]

    options = {"front alone": df["front_score"], "back alone": df["back_score"], "mean of both": df["score"],
               "highest of both": df[["front_score", "back_score"]].max(axis=1),
               "lowest of both": df[["front_score", "back_score"]].min(axis=1)}
    table = pd.DataFrame([{"score": name, **metrics.summary(y, values, MAX_FPR)} for name, values in options.items()])
    print(f"== Each side and the simple combinations ({int(y.sum())} positives, {int((~y).sum())} negatives)")
    print(table.to_string(index=False))

    flags = {side: df[f"{side}_score"] > metrics.threshold_at_fpr(y, df[f"{side}_score"], MAX_FPR)
             for side in ("front", "back")}
    overlap = pd.DataFrame({"caught by": ["both sides", "front only", "back only", "neither"]})
    for label, mask in (("positives", y), ("negatives", ~y)):
        overlap[label] = [int((flags["front"] & flags["back"] & mask).sum()), int((flags["front"] & ~flags["back"] & mask).sum()),
                          int((~flags["front"] & flags["back"] & mask).sum()), int((~flags["front"] & ~flags["back"] & mask).sum())]
    print(f"\n== Overlap of the two sides, each with its own threshold ({MAX_FPR:.0%} of the negatives)")
    print(overlap.to_string(index=False))

    threshold = metrics.threshold_at_fpr(y, df["score"], MAX_FPR)
    df["flagged"] = df["score"] > threshold
    tables = {"overall": table, "overlap": overlap,
              "by_fold": pd.DataFrame([{"fold": fold, **metrics.summary(part["positive"], part["score"], MAX_FPR)}
                                       for fold, part in df.groupby("fold")]),
              "by_channel": metrics.by_group(df, "channel", threshold),
              "by_label_source": metrics.by_group(df, "label_source", threshold).drop(columns=["roc_auc_inside_group"])}
    for name, title in (("by_fold", "Mean of both sides, by fold"),
                        ("by_channel", f"Mean of both sides, by channel (one threshold: {threshold:.3f})"),
                        ("by_label_source", "Mean of both sides, by label source")):
        print(f"\n== {title}")
        print(tables[name].to_string(index=False))

    rules_path = output_dir(load_config()) / "rules_per_customer.csv"
    df["rule_failed"] = False
    if rules_path.exists():
        rules = pd.read_csv(rules_path)
        failed = set(rules.loc[rules.filter(regex="^R_").eq("FAIL").any(axis=1), "anon_id"])
        df["rule_failed"] = df["anon_id"].isin(failed)
        rows = []
        for name, flagged in (("rules alone", df["rule_failed"]), ("classifier alone", df["flagged"]),
                              ("classifier or rules", df["flagged"] | df["rule_failed"])):
            rows.append({"flagged by": name, "positives_flagged": int((flagged & y).sum()),
                         "recall": round(float(flagged[y].mean()), 4), "negatives_flagged": int((flagged & ~y).sum()),
                         "false_positive_rate": round(float(flagged[~y].mean()), 4)})
        tables["with_rules"] = pd.DataFrame(rows)
        print("\n== Classifier and rules together (flagged when the score is over the threshold or a rule failed)")
        print(tables["with_rules"].to_string(index=False))
        print(f"   positives only the rules catch: {int((df['rule_failed'] & ~df['flagged'] & y).sum())}")

    run_dir = ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, part in tables.items():
        part.to_csv(run_dir / f"metrics_{name}.csv", index=False)
    df.to_csv(run_dir / "oof_predictions.csv", index=False)
    # Positives the combined score does not flag, lowest score first: the queue of the visual review
    # (streamlit run scripts/13_visual_review.py -- --queue runs/<name>/missed_positives.csv).
    missed = df[y & ~df["flagged"]].sort_values("score")
    missed[["anon_id", "label_source", "channel", "fold", "front_score", "back_score", "score", "rule_failed"]] \
        .to_csv(run_dir / "missed_positives.csv", index=False)
    print(f"\npositives the combined score does not flag: {len(missed)} "
          f"({int(missed['rule_failed'].sum())} of them fail a rule) -> {run_dir / 'missed_positives.csv'}")
    (run_dir / "config.json").write_text(json.dumps(
        {"side": "both", "front_runs": args.front, "back_runs": args.back, "combination": "mean",
         "threshold_at_2pct_fpr": threshold}, indent=2), encoding="utf-8")
    print(f"\nsaved in {run_dir}")


if __name__ == "__main__":
    main()
