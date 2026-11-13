"""Phase 3 · Split the people who have photos into five folds for the cross-validation.

It connects to nothing. It reads the manifest, the photos on disk and, when they exist, the rule
results (to keep the people who share a back together) and writes data/<stage>/splits.csv:
anon_id, fold, group. Rules of the split: id_fraud_detector/evaluation/splits.py.

The split is made once and everything else reads it. Run it again only when the dataset changes
(--force); the people who were already there keep their fold as long as their group is the same.

Usage:  python scripts/30_make_splits.py [--force] [--folds 5]
"""
import argparse
import sys

import pandas as pd

from id_fraud_detector.common import image_path, load_config, output_dir
from id_fraud_detector.evaluation.splits import assign_folds, person_groups, shared_back_clusters
from id_fraud_detector.ocr_store import SIDES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.csv")
    args = ap.parse_args()

    cfg = load_config()
    out = output_dir(cfg)
    target = out / "splits.csv"
    if target.exists() and not args.force:
        sys.exit(f"ERROR: {target} already exists. Use --force to make the split again.")

    manifest = pd.read_csv(out / "manifest.csv",
                           usecols=["anon_id", "tenant", "label", "label_source", "channel", "has_images"])
    people = manifest[manifest["has_images"]].copy()
    on_disk = [all(image_path(out, r.tenant, r.label, r.anon_id, side).exists() for side in SIDES)
               for r in people.itertuples(index=False)]
    print(f"people selected for images: {len(people)}   with both photos on disk: {sum(on_disk)}")
    people = people[on_disk].reset_index(drop=True)

    clusters = []
    rules, values = out / "rules_per_customer.csv", out / "private" / "rules_values.csv"
    if rules.exists() and values.exists():
        clusters = shared_back_clusters(pd.read_csv(rules), pd.read_csv(values, dtype=str))
        print(f"groups of people who share a back: {len(clusters)}")
    else:
        print("no rule results on disk: only the rings by label source are kept together")

    people["group"] = person_groups(people, clusters)
    people["stratum"] = people["label"] + "|" + people["channel"].fillna("(none)")
    people["fold"] = assign_folds(people, args.folds, cfg["seed"])
    people[["anon_id", "fold", "group"]].to_csv(target, index=False)

    sizes = people.groupby("group").size()
    print(f"groups with more than one person: {int((sizes > 1).sum())}"
          f"   people in them: {int(sizes[sizes > 1].sum())}   largest: {int(sizes.max())}")
    print("\nPeople per fold and label:")
    print(people.pivot_table(index="fold", columns="label", values="anon_id", aggfunc="size", fill_value=0).to_string())
    print("\nPositives per fold and label source:")
    positives = people[people["label"] == "POSITIVE"]
    print(positives.pivot_table(index="fold", columns="label_source", values="anon_id", aggfunc="size",
                                fill_value=0).to_string())
    print("\nPeople per fold, label and channel:")
    print(people.pivot_table(index="fold", columns=["label", "channel"], values="anon_id", aggfunc="size",
                             fill_value=0).to_string())
    print(f"\nsplit: {target}")


if __name__ == "__main__":
    main()
