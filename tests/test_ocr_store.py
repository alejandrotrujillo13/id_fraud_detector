"""OCR store tests: own text, and text borrowed from another tenant only for identical images.
No database and no S3: both are replaced by fakes.

Run:  python -m unittest discover tests
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from id_fraud_detector import ocr_store


def line(text):
    return {"Type": "LINE", "DetectedText": text, "Confidence": 99.0}


class FakeConnection:
    def close(self):
        pass


# Customers of the tenant being exported ("esmio"). Document numbers are made up.
OWN = pd.DataFrame([
    {"client_id": 1, "anon_id": "same_images", "document_number": "100",
     "front_id_image": "esmio/front_1.png", "back_id_image": "esmio/back_1.png"},
    {"client_id": 2, "anon_id": "back_retaken", "document_number": "200",
     "front_id_image": "esmio/front_2.png", "back_id_image": "esmio/back_2.png"},
    {"client_id": 3, "anon_id": "own_text", "document_number": "300",
     "front_id_image": "esmio/front_3.png", "back_id_image": "esmio/back_3.png"},
    {"client_id": 4, "anon_id": "only_here", "document_number": "400",
     "front_id_image": "esmio/front_4.png", "back_id_image": "esmio/back_4.png"},
    {"client_id": 5, "anon_id": "unreadable_object", "document_number": "500",
     "front_id_image": "esmio/front_5.png", "back_id_image": "esmio/back_5.png"},
])
OWN_OCR = {3: {"client_id": 3, "validation_updated_at": "2026-01-01",
               "front_lines": [line("own front")], "back_lines": [line("own back")]}}

# The same people in the other tenant ("llevaloya"), where the photos were taken.
OTHER = pd.DataFrame([
    {"client_id": 11, "document_number": "100", "front_id_image": "llevaloya/front_11.png",
     "back_id_image": "llevaloya/back_11.png"},
    {"client_id": 12, "document_number": "200", "front_id_image": "llevaloya/front_12.png",
     "back_id_image": "llevaloya/back_12.png"},
    {"client_id": 15, "document_number": "500", "front_id_image": "llevaloya/front_15.png",
     "back_id_image": "llevaloya/back_15.png"},
])
OTHER_OCR = {cid: {"client_id": cid, "validation_updated_at": "2025-05-05",
                   "front_lines": [line(f"front of {cid}")], "back_lines": [line(f"back of {cid}")]}
             for cid in (11, 12, 15)}

FINGERPRINTS = {
    "esmio/front_1.png": ("etag-a", 10), "llevaloya/front_11.png": ("etag-a", 10),
    "esmio/back_1.png": ("etag-b", 20), "llevaloya/back_11.png": ("etag-b", 20),
    "esmio/front_2.png": ("etag-c", 30), "llevaloya/front_12.png": ("etag-c", 30),
    "esmio/back_2.png": ("etag-d", 40), "llevaloya/back_12.png": ("etag-other", 41),   # back photo retaken
    "esmio/front_5.png": None, "llevaloya/front_15.png": None,                          # cannot be read
    "esmio/back_5.png": None, "llevaloya/back_15.png": None,
}


def fake_ocr_lines(rows):
    return lambda con, client_ids, chunk_size=300: (rows[int(c)] for c in client_ids if int(c) in rows)


class Borrowing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        with mock.patch.object(ocr_store, "fetch_ocr_lines", fake_ocr_lines(OWN_OCR)), mock.patch("builtins.print"):
            ocr_store.export_ocr(FakeConnection(), "esmio", OWN, self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def borrow(self):
        by_document = mock.Mock(side_effect=lambda con, documents: OTHER[OTHER["document_number"].isin(list(documents))])
        with mock.patch.object(ocr_store, "fetch_clients_by_document", by_document), \
                mock.patch.object(ocr_store, "fetch_ocr_lines", fake_ocr_lines(OTHER_OCR)), \
                mock.patch.object(ocr_store, "s3_fingerprint", lambda s3, bucket, key: FINGERPRINTS[key]):
            stats = ocr_store.borrow_ocr("esmio", OWN, "llevaloya", self.dir, lambda tenant: FakeConnection(),
                                         s3=None, bucket="bucket")
        return stats, by_document

    def flags(self):
        return ocr_store.ocr_flags(self.dir, OWN["anon_id"]).set_index("anon_id")

    def test_own_export_records_where_the_text_came_from(self):
        flags = self.flags()
        self.assertEqual(flags.loc["own_text", ["has_front_ocr", "front_ocr_from", "back_ocr_from"]].tolist(),
                         [True, "esmio", "esmio"])
        self.assertFalse(flags.loc["same_images", "has_back_ocr"])
        self.assertTrue(pd.isna(flags.loc["same_images", "back_ocr_from"]))

    def test_text_is_borrowed_only_for_identical_images(self):
        stats, _ = self.borrow()
        flags = self.flags()
        # both images identical in both tenants: both sides borrowed
        self.assertEqual(flags.loc["same_images", ["front_ocr_from", "back_ocr_from"]].tolist(), ["llevaloya"] * 2)
        self.assertEqual(ocr_store.lines(ocr_store.read(self.dir, "same_images"), "back")[0]["DetectedText"], "back of 11")
        # back photo differs: only the front is borrowed
        self.assertEqual(flags.loc["back_retaken", ["has_front_ocr", "has_back_ocr"]].tolist(), [True, False])
        # objects that cannot be read are never treated as identical
        self.assertEqual(flags.loc["unreadable_object", ["has_front_ocr", "has_back_ocr"]].tolist(), [False, False])
        # own text is never replaced; a person not present in the other tenant stays without text
        self.assertEqual(flags.loc["own_text", "back_ocr_from"], "esmio")
        self.assertFalse(flags.loc["only_here", "has_back_ocr"])
        self.assertEqual(stats, {"customers_missing_text": 4, "found_in_other_tenant": 3, "front_borrowed": 2,
                                 "back_borrowed": 1, "front_different_image": 1, "back_different_image": 2})

    def test_second_run_does_not_search_again(self):
        self.borrow()
        stats, by_document = self.borrow()
        self.assertEqual(stats["customers_missing_text"], 0)
        by_document.assert_not_called()


if __name__ == "__main__":
    unittest.main()
