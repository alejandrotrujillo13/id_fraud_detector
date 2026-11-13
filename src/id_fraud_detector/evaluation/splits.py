"""Five-fold split by person for the cross-validation of the classifier.

Rules (docs/plan.md §5.5):
  - every person is in exactly one fold;
  - a whole fraud ring stays in one fold, so the model that scores a ring never saw its template
    while training. Rings: the label sources listed in RING_OF, and the people who share a back
    (same sequence or serial under different ID numbers, i.e. R_REPEATED_BACK = FAIL);
  - folds are balanced in label and channel.

The split is deterministic: it depends only on the people, their groups and the seed.
"""
import hashlib

import pandas as pd

# Label sources that are one fraud ring. The four new FINANDINA customers go with the store-623
# ring: 64 of the 68 customers of that source were already positives of that window.
RING_OF = {"POSSIBLE_FRAUD_WINDOW": "ring_623", "CONFIRMED_FINANDINA_LIBRE_DESTINO": "ring_623",
           "CONFIRMED_AMORETT": "ring_amorett"}


def shared_back_clusters(rules: pd.DataFrame, values: pd.DataFrame) -> list[set]:
    """Sets of anon_ids that share a back: people whose R_REPEATED_BACK failed and have the same
    code sequence or the same serial. `rules` is rules_per_customer.csv, `values` is
    private/rules_values.csv."""
    failed = set(rules.loc[rules["R_REPEATED_BACK"] == "FAIL", "anon_id"])
    values = values[values["anon_id"].isin(failed)]
    clusters = []
    for column in ("code_sequence", "serial_digits"):
        for _, group in values.dropna(subset=[column]).groupby(column):
            if len(group) > 1:
                clusters.append(set(group["anon_id"]))
    return clusters


def person_groups(people: pd.DataFrame, clusters: list[set] = ()) -> pd.Series:
    """Group of each person (index = people.index). `people` needs anon_id and label_source.
    People of a ring, or linked by a shared back (directly or through others), get the same
    group; everybody else is alone in a group named after the person."""
    parent = {anon: anon for anon in people["anon_id"]}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(members):
        members = [m for m in members if m in parent]
        for other in members[1:]:
            parent[find(other)] = find(members[0])

    ring = people["label_source"].map(RING_OF)
    for _, members in people[ring.notna()].groupby(ring[ring.notna()]):
        union(list(members["anon_id"]))
    for cluster in clusters:
        union(sorted(cluster))

    root = people["anon_id"].map(find)
    names = {}
    for key, members in people.groupby(root):
        rings = sorted(set(members["label_source"].map(RING_OF).dropna()))
        names[key] = "+".join(rings) if rings else (f"shared_back_{key}" if len(members) > 1 else key)
    return root.map(names)


def assign_folds(people: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.Series:
    """Fold (0..n_folds-1) of each person. `people` needs anon_id, group, label and stratum
    (for example label|channel). Groups are placed whole, the largest first, each in the fold
    that keeps every stratum, and every label as a whole, most evenly spread. (Without the
    label, a fold that holds a ring of app positives would still get its full share of the
    other positives.)"""
    balanced = pd.concat([people["stratum"], "label:" + people["label"].astype(str)], ignore_index=True)
    groups = pd.concat([people["group"], people["group"]], ignore_index=True)
    totals = balanced.value_counts()
    per_group = pd.crosstab(groups, balanced).reindex(columns=totals.index, fill_value=0)
    tie_break = {group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest() for group in per_group.index}
    order = sorted(per_group.index, key=lambda group: (-int(per_group.loc[group].sum()), tie_break[group]))
    share = (per_group / totals).to_numpy()  # each group's share of every stratum
    position = {group: i for i, group in enumerate(per_group.index)}
    filled = [share[0] * 0 for _ in range(n_folds)]
    fold_of = {}
    for group in order:
        added = share[position[group]]
        # increase of the sum of squares if the group goes to each fold: smallest = most even
        cost = [float(((fold + added) ** 2 - fold ** 2).sum()) for fold in filled]
        best = min(range(n_folds), key=lambda fold: (round(cost[fold], 12), fold))
        filled[best] = filled[best] + added
        fold_of[group] = best
    return people["group"].map(fold_of)
