import tempfile
import unittest
from pathlib import Path

import pandas as pd

from id_fraud_detector.model.data import GENUINE_DOCUMENT, excluded_by_review


class ExcludedByReviewTest(unittest.TestCase):
    def test_only_the_ids_found_genuine_are_left_out(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.assertEqual(excluded_by_review(out), set())  # no review yet
            (out / "review").mkdir()
            pd.DataFrame({"anon_id": ["a", "b", "c"], "modality": [GENUINE_DOCUMENT, "unsure", "replaced_photo"],
                          "notes": ["", "", ""]}).to_csv(out / "review" / "modalities.csv", index=False)
            self.assertEqual(excluded_by_review(out), {"a"})


if __name__ == "__main__":
    unittest.main()
