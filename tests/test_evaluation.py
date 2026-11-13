"""Split and metric tests (numpy and pandas only, no GPU).

Run:  python -m unittest discover tests
"""
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from id_fraud_detector.evaluation import metrics
from id_fraud_detector.evaluation.splits import assign_folds, person_groups, shared_back_clusters
from id_fraud_detector.model.data import reduced


def people(n_negatives=400, n_fraud=60, n_ring=40, n_shop=10):
    rows = [{"anon_id": f"n{i:04d}", "label": "NEGATIVE", "label_source": "NO_FLAG_WITH_PAID_CREDIT",
             "channel": "APP" if i % 10 == 0 else "COMMERCE"} for i in range(n_negatives)]
    rows += [{"anon_id": f"f{i:04d}", "label": "POSITIVE", "label_source": "FRAUD",
              "channel": "APP" if i % 3 == 0 else "COMMERCE"} for i in range(n_fraud)]
    rows += [{"anon_id": f"r{i:04d}", "label": "POSITIVE", "label_source": "POSSIBLE_FRAUD_WINDOW", "channel": "APP"}
             for i in range(n_ring)]
    rows += [{"anon_id": f"a{i:04d}", "label": "POSITIVE", "label_source": "CONFIRMED_AMORETT", "channel": "APP"}
             for i in range(n_shop)]
    return pd.DataFrame(rows)


class Groups(unittest.TestCase):
    def test_rings_and_shared_backs_become_one_group(self):
        df = people()
        df.loc[len(df)] = {"anon_id": "x1", "label": "POSITIVE", "label_source": "CONFIRMED_FINANDINA_LIBRE_DESTINO",
                           "channel": "APP"}
        clusters = [{"f0001", "f0002"}, {"f0002", "f0003"}, {"f0010", "r0000"}, {"f0020", "not_in_the_split"}]
        group = person_groups(df, clusters).to_numpy()
        by_person = dict(zip(df["anon_id"], group))
        self.assertEqual({by_person[f"r{i:04d}"] for i in range(40)}, {"ring_623"})
        self.assertEqual(by_person["x1"], "ring_623")       # the new Finandina customers go with that ring
        self.assertEqual(by_person["f0010"], "ring_623")    # shares a back with someone of the ring
        self.assertEqual({by_person[f"a{i:04d}"] for i in range(10)}, {"ring_amorett"})
        self.assertEqual(len({by_person["f0001"], by_person["f0002"], by_person["f0003"]}), 1)  # linked through f0002
        self.assertTrue(by_person["f0001"].startswith("shared_back_"))
        self.assertEqual(by_person["f0020"], "f0020")       # alone: its partner is not among the people
        self.assertEqual(by_person["n0000"], "n0000")

    def test_shared_back_clusters_only_use_people_whose_rule_failed(self):
        rules = pd.DataFrame({"anon_id": list("abcde"), "R_REPEATED_BACK": ["FAIL", "FAIL", "PASS", "FAIL", "FAIL"]})
        values = pd.DataFrame({"anon_id": list("abcde"), "code_sequence": ["1", "1", "1", "2", None],
                               "serial_digits": [None, "9", "9", "8", "8"]})
        self.assertEqual(sorted(sorted(c) for c in shared_back_clusters(rules, values)), [["a", "b"], ["d", "e"]])


class Folds(unittest.TestCase):
    def setUp(self):
        self.df = people()
        self.df["group"] = person_groups(self.df)
        self.df["stratum"] = self.df["label"] + "|" + self.df["channel"]
        self.df["fold"] = assign_folds(self.df, 5, seed=42)

    def test_every_group_is_in_one_fold_and_folds_are_balanced(self):
        self.assertEqual(self.df.groupby("group")["fold"].nunique().max(), 1)
        self.assertEqual(sorted(self.df["fold"].unique()), [0, 1, 2, 3, 4])
        negatives = self.df[self.df["label"] == "NEGATIVE"].groupby("fold").size()
        self.assertTrue(negatives.between(70, 90).all(), negatives.to_dict())
        positives = self.df[self.df["label"] == "POSITIVE"].groupby("fold").size()
        # 110 positives with a ring of 40: its fold gets the ring and fewer of the other positives
        self.assertTrue(positives.between(14, 46).all(), positives.to_dict())
        ring_fold = self.df.loc[self.df["group"] == "ring_623", "fold"].iloc[0]
        others = self.df[(self.df["label"] == "POSITIVE") & (self.df["group"] != "ring_623")].groupby("fold").size()
        self.assertEqual(others.idxmin(), ring_fold, others.to_dict())
        app_negatives = self.df[self.df["stratum"] == "NEGATIVE|APP"].groupby("fold").size()
        self.assertTrue(app_negatives.between(6, 10).all(), app_negatives.to_dict())

    def test_the_split_is_reproducible_and_does_not_depend_on_row_order(self):
        shuffled = self.df.sample(frac=1, random_state=1).reset_index(drop=True)
        again = dict(zip(shuffled["anon_id"], assign_folds(shuffled, 5, seed=42)))
        self.assertEqual(again, dict(zip(self.df["anon_id"], self.df["fold"])))


class Metrics(unittest.TestCase):
    def test_known_values(self):
        y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
        self.assertAlmostEqual(metrics.roc_auc(y, s), 0.75)
        self.assertAlmostEqual(metrics.average_precision(y, s), 0.8333, places=4)
        self.assertAlmostEqual(metrics.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)  # ties count half
        self.assertAlmostEqual(metrics.average_precision([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertEqual(metrics.roc_auc([1, 1], [0.2, 0.3]) != metrics.roc_auc([1, 1], [0.2, 0.3]), True)  # nan

    def test_recall_at_a_fixed_false_positive_rate(self):
        rng = np.random.default_rng(0)
        negatives, positives = rng.normal(0, 1, 1000), rng.normal(3, 1, 100)
        y = np.r_[np.zeros(1000), np.ones(100)]
        s = np.r_[negatives, positives]
        threshold = metrics.threshold_at_fpr(y, s, 0.02)
        self.assertEqual(int((negatives > threshold).sum()), 20)  # exactly 2 % of the negatives
        self.assertAlmostEqual(metrics.recall_at_fpr(y, s, 0.02), float((positives > threshold).mean()))
        # with very few negatives none may be flagged: the threshold is the highest negative
        self.assertEqual(metrics.threshold_at_fpr([0, 0, 1], [0.1, 0.2, 0.9], 0.02), 0.2)
        self.assertEqual(metrics.recall_at_fpr([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.15], 0.02), 0.5)

    def test_by_group_uses_the_global_threshold(self):
        df = pd.DataFrame({"positive": [True, True, False, False, True, False],
                           "score": [0.9, 0.2, 0.1, 0.6, 0.7, 0.3], "channel": ["APP"] * 4 + ["COMMERCE"] * 2})
        table = metrics.by_group(df, "channel", threshold=0.5).set_index("channel")
        self.assertEqual(table.loc["APP", ["positives_flagged", "negatives_flagged"]].tolist(), [1, 1])
        self.assertEqual(table.loc["COMMERCE", "recall"], 1.0)
        self.assertEqual(table.loc["APP", "roc_auc_inside_group"], 0.75)


class ReducedPhoto(unittest.TestCase):
    def test_the_photo_covers_the_input_without_bars_or_distortion(self):
        wide = Image.new("RGB", (1280, 720), (10, 200, 30))   # 16:9: the sides are cut
        tall = Image.new("RGB", (640, 480), (10, 200, 30))    # 4:3: top and bottom are cut
        for image in (wide, tall):
            array = reduced(image, (512, 320))
            self.assertEqual(array.shape, (320, 512, 3))
            self.assertTrue((array == np.array([10, 200, 30])).all())  # no padding color anywhere


if __name__ == "__main__":
    unittest.main()
