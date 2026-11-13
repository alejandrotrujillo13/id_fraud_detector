"""Phase 5 · Calibrate the classifier and build the 0-100 risk indicator.

It connects to nothing. It reads the out-of-fold scores of one front run and one back run of
31_train.py (each person scored by a model that never saw them), the manifest and the result of
the rules, and:

  1. gives each negative with photos the weight of the negatives it stands for, so that every
     figure speaks about the labelled population and not about the training sample;
  2. calibrates the mean of the two scores into a probability of forgery (Platt and isotonic
     are compared; the one asked with --method is kept);
  3. measures the weight of each rule (rules/ and risk.rule_weight) and applies it;
  4. reports how well the indicator is calibrated and what each cut point would send to review.

Everything reported is "nested": the calibration and the rule weights applied to a fold were
fitted without that fold. The parameters saved for use with new customers are fitted with
everyone. People the visual review found to hold a genuine document are left out.

Saved in runs/<name>/: calibration.json (parameters), indicator_oof.csv (one row per person),
reliability.csv and cut_points.csv.

Usage:  python scripts/50_calibrate.py --front runs/<front run> --back runs/<back run> [--name indicator]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from id_fraud_detector import risk
from id_fraud_detector.common import ROOT, load_config, output_dir
from id_fraud_detector.evaluation import metrics
from id_fraud_detector.evaluation.population import sampling_weights, weighted_share_above, weighted_threshold
from id_fraud_detector.model.data import excluded_by_review

REVIEW_SHARE = 0.02  # at most this share of the legitimate customers goes to manual review (docs/plan.md §4)
METHODS = {"platt": (risk.fit_platt, risk.apply_platt), "isotonic": (risk.fit_isotonic, risk.apply_isotonic)}


def load_scores(front: Path, back: Path) -> pd.DataFrame:
    sides = {}
    for side, run in (("front", front), ("back", back)):
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config["side"] != side:
            raise SystemExit(f"ERROR: {run} is a run of the {config['side']}, not of the {side}.")
        sides[side] = pd.read_csv(run / "oof_predictions.csv")
    df = sides["front"].rename(columns={"score": "front_score"}).merge(
        sides["back"][["anon_id", "fold", "score"]].rename(columns={"score": "back_score"}),
        on=["anon_id", "fold"], how="inner", validate="one_to_one")
    if len(df) != len(sides["front"]) or len(df) != len(sides["back"]):
        raise SystemExit("ERROR: the two runs do not hold the same people in the same folds.")
    df["score"] = df[["front_score", "back_score"]].mean(axis=1)
    return df


def rule_counts(df: pd.DataFrame, rules: list[str]) -> pd.DataFrame:
    rows = []
    for rule in rules:
        row = {"rule": rule}
        for name, mask in (("forged", df["positive"]), ("genuine", ~df["positive"])):
            row[f"evaluated_{name}"] = int(df.loc[mask, rule].isin(["PASS", "FAIL"]).sum())
            row[f"failed_{name}"] = int((df.loc[mask, rule] == "FAIL").sum())
        row["weight"] = round(risk.rule_weight(row["failed_forged"], row["evaluated_forged"],
                                               row["failed_genuine"], row["evaluated_genuine"]), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def rule_factor(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    factor = np.ones(len(df))
    for rule, weight in weights.items():
        factor *= np.where(df[rule] == "FAIL", weight, 1.0)
    return factor


def losses(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": round(float(-(w * np.where(y, np.log(p), np.log(1 - p))).sum() / w.sum()), 5),
            "brier": round(float((w * (p - y) ** 2).sum() / w.sum()), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True, help="run of the front (31_train.py)")
    ap.add_argument("--back", required=True, help="run of the back (31_train.py)")
    ap.add_argument("--method", default="isotonic", choices=sorted(METHODS), help="calibration kept for the indicator")
    ap.add_argument("--name", default="indicator", help="name of the folder under runs/ for the result")
    args = ap.parse_args()

    out = output_dir(load_config())
    df = load_scores(Path(args.front), Path(args.back))
    left_out = df["anon_id"].isin(excluded_by_review(out)) & df["positive"]
    df = df[~left_out].reset_index(drop=True)
    manifest = pd.read_csv(out / "manifest.csv", usecols=["anon_id", "label", "channel", "registered_date"])
    manifest = manifest[~manifest["anon_id"].isin(excluded_by_review(out))]
    df["weight"] = sampling_weights(manifest, df["anon_id"]).to_numpy()
    y, w, folds = df["positive"].to_numpy(), df["weight"].to_numpy(), df["fold"].to_numpy()
    prevalence = float(w[y].sum() / w.sum())
    print(f"people with photos: {len(df)}  positives: {int(y.sum())}  (left out by the visual review: {int(left_out.sum())})")
    print(f"labelled population they stand for: {w.sum():,.0f}  ({w[~y].sum():,.0f} negatives; forged {prevalence:.2%}; "
          f"in the sample {y.mean():.2%})")

    print(f"\n== What 'at most {REVIEW_SHARE:.0%} of the negatives' means, on the classifier score alone")
    sample_cut = metrics.threshold_at_fpr(y, df["score"], REVIEW_SHARE)
    population_cut = weighted_threshold(df.loc[~y, "score"], w[~y], REVIEW_SHARE)
    for name, cut in (("counting each sampled negative once", sample_cut), ("weighting to the population", population_cut)):
        print(f"{name:38s} threshold {cut:.3f}  positives caught {int((df['score'] > cut)[y].sum())}/{int(y.sum())} "
              f"= {(df['score'] > cut)[y].mean():.3f}  negatives of the sample {(df['score'] > cut)[~y].mean():.4f}  "
              f"of the population {weighted_share_above(df.loc[~y, 'score'], w[~y], cut):.4f}")

    print("\n== Calibration of the classifier score (nested; weighted to the population; lower is better)")
    nested = {name: np.zeros(len(df)) for name in METHODS}
    for fold in np.unique(folds):
        rest, held = folds != fold, folds == fold
        for name, (fit, apply) in METHODS.items():
            nested[name][held] = apply(fit(df.loc[rest, "score"], y[rest], w[rest]), df.loc[held, "score"])
    options = {"always the share of forged IDs": np.full(len(df), prevalence), "the raw score as it is": df["score"].to_numpy(),
               **{f"{name} calibration": values for name, values in nested.items()}}
    print(pd.DataFrame([{"probability": name, **losses(y, p, w)} for name, p in options.items()]).to_string(index=False))

    print("\n== Platt calibration fitted on the scores of each fold's model alone (do the five models agree?)")
    marks = (0.3, 0.5, 0.7, 0.9)
    rows = []
    for fold in [*np.unique(folds), "all"]:
        part = np.ones(len(df), bool) if fold == "all" else folds == fold
        params = risk.fit_platt(df.loc[part, "score"], y[part], w[part])
        rows.append({"model of fold": fold, "positives": int(y[part].sum()), "slope": round(params[0], 3),
                     "intercept": round(params[1], 3),
                     **{f"score {m}": round(float(risk.apply_platt(params, [m])[0]), 3) for m in marks}})
    print(pd.DataFrame(rows).to_string(index=False))

    rules_path = out / "rules_per_customer.csv"
    rule_names, final_weights, counts = [], {}, pd.DataFrame()
    df["factor"] = 1.0
    if rules_path.exists():
        results = pd.read_csv(rules_path)
        rule_names = [c for c in results.columns if c.startswith("R_")]
        df = df.merge(results[["anon_id", *rule_names]], on="anon_id", how="left", validate="one_to_one")
        df[rule_names] = df[rule_names].fillna("NOT_EVALUABLE")
        counts = rule_counts(df, rule_names)
        final_weights = {r.rule: r.weight for r in counts.itertuples() if r.weight > 1}
        print("\n== Rules: failures among the people evaluated, and the weight of a FAIL (everyone)")
        print(counts.to_string(index=False))
        factor, per_fold = np.ones(len(df)), []
        for fold in np.unique(folds):
            rest, held = folds != fold, folds == fold
            fold_counts = rule_counts(df[rest], rule_names)
            fold_weights = {r.rule: r.weight for r in fold_counts.itertuples() if r.weight > 1}
            factor[held] = rule_factor(df[held], fold_weights)
            per_fold.append({"held-out fold": fold, **{rule: fold_weights.get(rule, 1.0) for rule in final_weights}})
        df["factor"] = factor
        print("weights measured without each fold (the ones applied to it below):")
        print(pd.DataFrame(per_fold).to_string(index=False))

    df["probability"] = nested[args.method]
    df["classifier_indicator"] = risk.indicator(df["probability"])
    df["indicator"] = risk.indicator(df["probability"], df["factor"])
    ind = df["indicator"].to_numpy()
    print(f"\n== Indicator 0-100 ({args.method} calibration, then the rules; nested)")
    print(pd.DataFrame([{"probability": name, **losses(y, values / 100, w)} for name, values in
                        (("classifier alone", df["classifier_indicator"].to_numpy()), ("classifier and rules", ind))])
          .to_string(index=False))

    edges = [0, 0.5, 1, 2, 5, 10, 20, 40, 70, 90, 100.0001]
    df["band"] = pd.cut(ind, edges, right=False)
    reliability = df.groupby("band", observed=True).apply(lambda part: pd.Series({
        "people_with_photos": len(part), "positives": int(part["positive"].sum()),
        "population": round(part["weight"].sum()), "mean_indicator": round(float(np.average(part["indicator"], weights=part["weight"])), 2),
        "forged_per_100": round(100 * part.loc[part["positive"], "weight"].sum() / part["weight"].sum(), 2)}), include_groups=False)
    print("\nIs it calibrated? In each band, the mean indicator should be close to the forged IDs per 100 people:")
    print(reliability.to_string())

    review_cut = weighted_threshold(ind[~y], w[~y], REVIEW_SHARE)
    print(f"\n== Cut points (the indicator that sends {REVIEW_SHARE:.0%} of the legitimate customers to review is {review_cut:.1f})")
    rows = []
    for cut in sorted({round(review_cut, 1), 5, 10, 20, 30, 40, 50, 60, 70, 80, 90}):
        above = ind > cut
        caught, legit = int(above[y].sum()), float(w[above & ~y].sum())
        rows.append({"indicator_above": cut, "legit_flagged_pct": round(100 * legit / w[~y].sum(), 2),
                     "legit_flagged_in_sample": int((above & ~y).sum()), "forged_caught": caught,
                     "forged_caught_pct": round(100 * caught / y.sum(), 1),
                     "forged_per_100_flagged": round(100 * caught / (caught + legit), 1) if caught + legit else float("nan")})
    cut_points = pd.DataFrame(rows)
    print(cut_points.to_string(index=False))

    print(f"\n== By channel, indicator above {review_cut:.1f}")
    df["flagged"] = ind > review_cut
    by_channel = df.groupby("channel", dropna=False).apply(lambda part: pd.Series({
        "positives": int(part["positive"].sum()), "caught": int((part["flagged"] & part["positive"]).sum()),
        "negatives_with_photos": int((~part["positive"]).sum()),
        "legit_flagged_pct": round(100 * part.loc[part["flagged"] & ~part["positive"], "weight"].sum()
                                   / max(part.loc[~part["positive"], "weight"].sum(), 1e-9), 2)}), include_groups=False)
    print(by_channel.to_string())
    if rule_names:
        only_rules = (df["factor"] > 1) & df["flagged"] & ~(df["classifier_indicator"] > review_cut)
        print(f"flagged only because a rule failed: positives {int((only_rules & df['positive']).sum())}, "
              f"negatives {int((only_rules & ~df['positive']).sum())}")

    fit, _ = METHODS[args.method]
    final = fit(df["score"], y, w)
    run_dir = ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    calibration = {"front_run": args.front, "back_run": args.back, "combination": "mean", "method": args.method,
                   "parameters": ({"slope": final[0], "intercept": final[1]} if args.method == "platt"
                                  else {"scores": final[0].tolist(), "probabilities": final[1].tolist()}),
                   "rule_weights": final_weights, "review_cut_at_2pct_legit": review_cut,
                   "population": {"people": round(float(w.sum())), "forged_share": prevalence,
                                  "people_with_photos": len(df), "positives": int(y.sum()),
                                  "left_out_by_review": int(left_out.sum())}}
    (run_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    df[["anon_id", "fold", "label_source", "channel", "positive", "weight", "front_score", "back_score", "score",
        "probability", "factor", "classifier_indicator", "indicator"]].to_csv(run_dir / "indicator_oof.csv", index=False)
    reliability.reset_index().astype({"band": str}).to_csv(run_dir / "reliability.csv", index=False)
    cut_points.to_csv(run_dir / "cut_points.csv", index=False)
    if len(counts):
        counts.to_csv(run_dir / "rule_weights.csv", index=False)
    print(f"\nparameters for new customers (fitted with everyone): {calibration['parameters'] if args.method == 'platt' else 'isotonic curve'}"
          f"   rule weights: {final_weights}")
    print(f"saved in {run_dir}")


if __name__ == "__main__":
    main()
