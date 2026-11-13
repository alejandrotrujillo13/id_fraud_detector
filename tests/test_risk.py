import unittest

import numpy as np
import pandas as pd

from id_fraud_detector import risk
from id_fraud_detector.evaluation.population import sampling_weights, weighted_share_above, weighted_threshold


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.truth = rng.uniform(0.02, 0.98, 20000)
        self.labels = rng.uniform(size=20000) < self.truth
        self.scores = risk.sigmoid(3 * risk.logit(self.truth) + 1)  # an overconfident, shifted score

    def test_platt_recovers_the_probability_behind_a_distorted_score(self):
        params = risk.fit_platt(self.scores, self.labels)
        self.assertAlmostEqual(params[0], 1 / 3, delta=0.02)
        self.assertLess(np.abs(risk.apply_platt(params, self.scores) - self.truth).mean(), 0.02)

    def test_isotonic_is_monotone_and_close_to_the_truth(self):
        curve = risk.fit_isotonic(self.scores, self.labels)
        self.assertTrue((np.diff(curve[1]) >= 0).all())
        self.assertLess(np.abs(risk.apply_isotonic(curve, self.scores) - self.truth).mean(), 0.03)

    def test_weights_move_the_calibration_to_the_population(self):
        # Negatives that stand for 5 people each: the probability of the same score goes down.
        weights = np.where(self.labels, 1.0, 5.0)
        plain = risk.apply_isotonic(risk.fit_isotonic(self.scores, self.labels), [0.5])[0]
        weighted = risk.apply_isotonic(risk.fit_isotonic(self.scores, self.labels, weights), [0.5])[0]
        self.assertLess(weighted, plain)


class RuleWeightTest(unittest.TestCase):
    def test_a_rule_that_never_failed_weighs_one(self):
        self.assertEqual(risk.rule_weight(0, 130, 0, 990), 1.0)

    def test_two_failures_are_not_enough_for_a_large_weight(self):
        self.assertLess(risk.rule_weight(2, 126, 0, 929), 3)

    def test_many_failures_only_among_forged_ids_weigh_a_lot(self):
        weight = risk.rule_weight(37, 137, 0, 990)
        self.assertGreater(weight, 50)
        self.assertLess(weight, (37 / 137) / (0.5 / 990))  # below the naive ratio: it is the cautious end

    def test_indicator_multiplies_the_odds(self):
        self.assertAlmostEqual(float(risk.indicator(0.5)), 50.0)
        self.assertAlmostEqual(float(risk.indicator(0.01, 99.0)), 50.0, places=4)
        self.assertEqual(float(risk.indicator(0.2, 1.0)), 20.0)


class PopulationTest(unittest.TestCase):
    def test_negatives_weigh_what_they_stand_for_and_positives_one(self):
        manifest = pd.DataFrame({
            "anon_id": list("abcdefg"),
            "label": ["NEGATIVE"] * 6 + ["POSITIVE"],
            "channel": ["APP", "APP", "COMMERCE", "COMMERCE", "COMMERCE", "COMMERCE", "APP"],
            "registered_date": ["2026-01-05", "2026-01-20", "2026-01-02", "2026-01-03", "2026-01-04", "2026-02-01", "2026-01-09"]})
        weights = sampling_weights(manifest, ["a", "b", "c", "f", "g"])
        self.assertEqual(weights.to_dict(), {"a": 1.0, "b": 1.0, "c": 3.0, "f": 1.0, "g": 1.0})
        self.assertEqual(weights[["a", "b", "c", "f"]].sum(), 6)  # the six negatives of the manifest
        with self.assertRaises(ValueError):
            sampling_weights(manifest, ["a", "b", "c", "g"])  # nobody sampled in COMMERCE of February

    def test_weighted_threshold_leaves_at_most_the_share_above(self):
        scores, weights = np.array([0.9, 0.8, 0.5, 0.2, 0.1]), np.array([1.0, 1.0, 2.0, 3.0, 3.0])
        cut = weighted_threshold(scores, weights, 0.2)
        self.assertEqual(cut, 0.5)
        self.assertAlmostEqual(weighted_share_above(scores, weights, cut), 0.2)
        self.assertEqual(weighted_share_above(scores, weights, weighted_threshold(scores, weights, 0.05)), 0.0)
        self.assertEqual(weighted_share_above(scores, weights, weighted_threshold(scores, weights, 1.0)), 1.0)


if __name__ == "__main__":
    unittest.main()
