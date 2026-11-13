import unittest

import numpy as np

from id_fraud_detector.evaluation.faces import chance_rate, chance_threshold, face_groups, pairs_above


def unit(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class FaceMatchingTest(unittest.TestCase):
    def test_pairs_above_finds_only_the_similar_pair(self):
        embeddings = unit([[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(pairs_above(embeddings, 0.9).tolist(), [[0, 1]])
        self.assertEqual(len(pairs_above(embeddings, 0.999)), 0)

    def test_groups_join_people_through_a_chain_of_matches(self):
        groups = face_groups(6, np.array([[0, 1], [1, 4], [2, 3]]))
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[4])
        self.assertEqual(groups[2], groups[3])
        self.assertEqual(len(set(groups.tolist())), 3)  # {0, 1, 4}, {2, 3}, {5}
        self.assertEqual(face_groups(3, np.empty((0, 2), dtype=int)).tolist(), [0, 1, 2])

    def test_chance_threshold_leaves_the_asked_share_of_pairs_above(self):
        rng = np.random.default_rng(0)
        embeddings = unit(rng.normal(size=(200, 16)))
        threshold = chance_threshold(embeddings, 0.01)
        self.assertAlmostEqual(chance_rate(embeddings, threshold), 0.01, delta=0.001)
        self.assertEqual(chance_rate(embeddings, 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
