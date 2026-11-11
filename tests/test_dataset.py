"""Labeling and sampling tests on a synthetic customer table (no database needed).

Run:  python -m unittest discover tests
"""
import unittest
from datetime import date

import pandas as pd

from id_fraud_detector.dataset import label_clients, sample_negatives, stored_ocr
from id_fraud_detector.rules.ocr_parse import ocr_lines

CFG = {"possible_fraud_windows": [{"tenant": "esmio", "start": "2026-09-13", "end": "2026-10-04"}]}


def clients(rows):
    """rows: (fraud_status, has_paid, registered_date, has_both_images)"""
    return pd.DataFrame([{
        "client_id": i, "document_number": str(1000 + i), "fraud_status": status, "channel": "APP",
        "registered_date": registered, "has_both_images": images, "has_paid": paid, "has_novated": False,
    } for i, (status, paid, registered, images) in enumerate(rows)])


class Labeling(unittest.TestCase):
    def setUp(self):
        self.df = clients([
            ("FRAUD", False, date(2025, 5, 1), True),            # 0 positive
            ("FRAUD", True, date(2025, 5, 1), True),             # 1 fraud that paid: excluded
            ("POSSIBLE FRAUD", False, date(2026, 9, 13), True),  # 2 window start (inclusive)
            ("POSSIBLE FRAUD", False, date(2026, 10, 4), True),  # 3 window end (inclusive)
            ("POSSIBLE FRAUD", False, date(2026, 10, 5), True),  # 4 after the window
            ("POSSIBLE FRAUD", True, date(2026, 9, 20), True),   # 5 in the window but paid
            (None, True, date(2025, 8, 1), True),                # 6 negative
            (None, False, date(2025, 8, 1), True),               # 7 no flag, never paid: unused
            ("VERIFICATION REQUIRED", True, date(2025, 8, 1), True),  # 8 other status
            ("FRAUD", False, date(2025, 5, 1), False),           # 9 positive group, but one image missing
        ])

    def test_esmio_labels(self):
        out = label_clients(self.df, "esmio", CFG)
        self.assertEqual(list(out["label"]),
                         ["POSITIVE", None, "POSITIVE", "POSITIVE", None, None, "NEGATIVE", None, None, None])
        self.assertEqual(list(out["label_source"][[0, 2, 6]]),
                         ["FRAUD", "POSSIBLE_FRAUD_WINDOW", "NO_FLAG_WITH_PAID_CREDIT"])
        self.assertEqual(list(out["group"]),
                         ["POSITIVE", "EXCLUDED_FRAUD_WITH_PAID", "EXCLUDED_OTHER_STATUS", "EXCLUDED_OTHER_STATUS",
                          "EXCLUDED_OTHER_STATUS", "EXCLUDED_OTHER_STATUS", "NEGATIVE", "UNUSED",
                          "EXCLUDED_OTHER_STATUS", "POSITIVE"])

    def test_window_only_applies_to_its_tenant(self):
        out = label_clients(self.df, "llevaloya", CFG)
        self.assertEqual(list(out["label"]),
                         ["POSITIVE", None, None, None, None, None, "NEGATIVE", None, None, None])

    def test_empty_table(self):
        out = label_clients(self.df.iloc[0:0], "esmio", CFG)
        self.assertEqual(len(out), 0)


class Sampling(unittest.TestCase):
    def setUp(self):
        rows = [(None, True, date(2025, 1 + i % 12, 1), True) for i in range(600)]
        self.df = label_clients(clients(rows), "esmio", CFG)

    def test_size_is_close_to_target_and_reproducible(self):
        first = sample_negatives(self.df, 120, seed=42)
        self.assertEqual(first, sample_negatives(self.df, 120, seed=42))
        self.assertEqual(len(first), 120)

    def test_every_month_is_represented(self):
        chosen = self.df[self.df["client_id"].isin(sample_negatives(self.df, 24, seed=1))]
        self.assertEqual(pd.to_datetime(chosen["registered_date"]).dt.month.nunique(), 12)

    def test_asking_for_more_than_available_returns_all(self):
        self.assertEqual(len(sample_negatives(self.df, 10_000, seed=1)), 600)


class OcrFile(unittest.TestCase):
    def test_exported_file_is_readable_by_the_parser(self):
        lines = [{"Type": "LINE", "DetectedText": "SEXO M", "Confidence": 99.1}]
        self.assertEqual(ocr_lines(stored_ocr(None, lines)["data_back_id"]), ["SEXO M"])
        self.assertEqual(ocr_lines(stored_ocr(None, None)["data_front_id"]), [])


if __name__ == "__main__":
    unittest.main()
