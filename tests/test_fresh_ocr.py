"""Fresh OCR tests: what is kept from Rekognition, resuming, and the account guard.
No AWS call: S3, Rekognition and STS are replaced by fakes.

Run:  python -m unittest discover tests
"""
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from id_fraud_detector import common, fresh_ocr, ocr_store
from id_fraud_detector.rules.ocr_parse import ocr_lines

KEYS = {"front": "tenant/front_1.png", "back": "tenant/back_1.png"}


def png(size: int = 64) -> bytes:
    noise = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noise).save(buffer, "PNG")
    return buffer.getvalue()


class FakeS3:
    def __init__(self, missing=()):
        self.missing, self.requested = set(missing), []

    def get_object(self, Bucket, Key):
        self.requested.append(Key)
        if Key in self.missing:
            raise RuntimeError("NoSuchKey (simulated)")
        return {"Body": io.BytesIO(png())}


class FakeRekognition:
    def __init__(self):
        self.calls = 0

    def detect_text(self, Image):
        self.calls += 1
        box = {"Width": 0.123456, "Height": 0.05, "Left": 0.1, "Top": 0.2}
        return {"TextModelVersion": "3.0", "ResponseMetadata": {"RequestId": "abc"}, "TextDetections": [
            {"Type": "LINE", "DetectedText": "REPUBLICA DE COLOMBIA", "Confidence": 99.123, "Id": 0,
             "Geometry": {"BoundingBox": box, "Polygon": [{"X": 0.1, "Y": 0.2}]}},
            {"Type": "WORD", "DetectedText": "REPUBLICA", "Confidence": 99.5, "Id": 1, "ParentId": 0,
             "Geometry": {"BoundingBox": box}},
        ]}


class WhatIsKept(unittest.TestCase):
    def test_every_detection_is_kept_as_rekognition_sent_it(self):
        response = FakeRekognition().detect_text(Image=None)
        self.assertEqual(fresh_ocr.text_detections(response), response["TextDetections"])
        self.assertEqual(fresh_ocr.text_detections({}), [])

    def test_the_saved_file_has_lines_words_geometry_and_model_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_ocr.run_person(FakeS3(), FakeRekognition(), "bucket", Path(tmp), "person", KEYS)
            payload = ocr_store.read(Path(tmp), "person")
        saved = ocr_store.lines(payload, "back")
        self.assertEqual(saved, FakeRekognition().detect_text(Image=None)["TextDetections"])
        self.assertEqual([d["Type"] for d in saved], ["LINE", "WORD"])
        self.assertEqual(payload["back"]["model_version"], "3.0")
        self.assertEqual(ocr_lines(payload["data_back_id"]), ["REPUBLICA DE COLOMBIA"])  # words are not read twice

    def test_small_images_are_sent_as_they_are(self):
        body = png()
        self.assertEqual(fresh_ocr.fit_for_rekognition(body), (body, "original"))

    def test_images_over_the_limit_are_sent_as_jpeg(self):
        body = png(200)
        with mock.patch.object(fresh_ocr, "MAX_BYTES", len(body) - 1):
            sent, how = fresh_ocr.fit_for_rekognition(body)
            self.assertTrue(how.startswith("jpeg_"))
            self.assertLessEqual(len(sent), fresh_ocr.MAX_BYTES)
        with Image.open(io.BytesIO(sent)) as image:
            self.assertEqual((image.format, image.size), ("JPEG", (200, 200)))


class EnlargedTextArea(unittest.TestCase):
    def line(self, left, top, width, height):
        return {"Type": "LINE", "Geometry": {"BoundingBox": {"Left": left, "Top": top, "Width": width, "Height": height}}}

    def test_the_text_area_is_cut_out_with_a_margin_and_enlarged(self):
        detections = [self.line(0.25, 0.25, 0.5, 0.1), self.line(0.30, 0.60, 0.3, 0.1),
                      {"Type": "WORD", "Geometry": {"BoundingBox": {"Left": 0, "Top": 0, "Width": 1, "Height": 1}}}]
        body, area = fresh_ocr.enlarged_text_area(png(200), detections, margin=0.05, scale=2)
        self.assertEqual(area, {"box": [40, 40, 160, 150], "scale": 2.0, "size": [240, 220]})  # words are ignored
        with Image.open(io.BytesIO(body)) as image:
            self.assertEqual((image.format, image.size), ("PNG", (240, 220)))

    def test_the_area_never_leaves_the_photo_and_never_exceeds_the_largest_side(self):
        body, area = fresh_ocr.enlarged_text_area(png(200), [self.line(-0.1, 0.9, 1.3, 0.3)], scale=4, max_side=300)
        self.assertEqual(area["box"], [0, 168, 200, 200])
        self.assertEqual(max(area["size"]), 300)

    def test_a_photo_without_text_has_nothing_to_cut_out(self):
        self.assertIsNone(fresh_ocr.enlarged_text_area(png(), []))


class Resuming(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_person(self, s3, rekognition, **options):
        """{side: error or None} for the sides that were tried."""
        tried = fresh_ocr.run_person(s3, rekognition, "bucket", self.dir, "person", KEYS, **options)
        return {side: info["error"] for side, info in tried.items()}

    def test_photos_are_kept_only_when_asked_and_then_never_downloaded_again(self):
        paths = {side: self.dir / "images" / f"person_{side}.png" for side in KEYS}
        self.run_person(FakeS3(), FakeRekognition(), paths=paths)
        self.assertFalse(paths["front"].exists())

        (self.dir / "person.json").unlink()  # start over, this time keeping the photos
        self.run_person(FakeS3(), FakeRekognition(), paths=paths, keep=True)
        self.assertEqual(paths["back"].read_bytes(), png())
        first = ocr_store.read(self.dir, "person")["back"]["image_md5"]

        (self.dir / "person.json").unlink()  # the OCR has to be redone: the photos come from disk
        s3 = FakeS3()
        self.assertEqual(self.run_person(s3, FakeRekognition(), paths=paths), {"front": None, "back": None})
        self.assertEqual(s3.requested, [])
        self.assertEqual(ocr_store.read(self.dir, "person")["back"]["image_md5"], first)

    def test_file_is_readable_by_the_rule_parser_and_tied_to_the_image(self):
        self.assertEqual(fresh_ocr.pending_sides(self.dir, "person"), ["front", "back"])
        self.assertEqual(self.run_person(FakeS3(), FakeRekognition()), {"front": None, "back": None})
        payload = ocr_store.read(self.dir, "person")
        self.assertEqual(ocr_lines(payload["data_back_id"]), ["REPUBLICA DE COLOMBIA"])
        self.assertEqual(len(payload["back"]["image_md5"]), 32)
        self.assertEqual(payload["back"]["image_bytes"], len(png()))
        self.assertEqual(fresh_ocr.pending_sides(self.dir, "person"), [])
        self.assertEqual(fresh_ocr.flags(self.dir, "person"),
                         {"front_done": True, "front_text": True, "back_done": True, "back_text": True})

    def test_answered_sides_are_not_sent_again(self):
        rekognition = FakeRekognition()
        self.run_person(FakeS3(), rekognition)
        s3 = FakeS3()
        self.assertEqual(self.run_person(s3, rekognition), {})
        self.assertEqual((rekognition.calls, s3.requested), (2, []))

    def test_a_failed_side_is_recorded_and_tried_again(self):
        rekognition = FakeRekognition()
        tried = self.run_person(FakeS3(missing=[KEYS["back"]]), rekognition)
        self.assertIsNone(tried["front"])
        self.assertIn("NoSuchKey", tried["back"])
        self.assertEqual(fresh_ocr.pending_sides(self.dir, "person"), ["back"])
        self.assertEqual(fresh_ocr.flags(self.dir, "person")["back_done"], False)
        s3 = FakeS3()
        self.assertEqual(self.run_person(s3, rekognition), {"back": None})
        self.assertEqual((rekognition.calls, s3.requested), (2, [KEYS["back"]]))  # the front was not paid twice

    def test_a_file_cut_short_counts_as_pending(self):
        (self.dir / "person.json").write_text('{"data_front_id": {"detec', encoding="utf-8")
        self.assertEqual(fresh_ocr.pending_sides(self.dir, "person"), ["front", "back"])
        self.assertEqual(fresh_ocr.flags(self.dir, "person")["front_done"], False)
        self.assertEqual(self.run_person(FakeS3(), FakeRekognition()), {"front": None, "back": None})


class PartialRun(unittest.TestCase):
    def people(self, order):
        import pandas as pd
        rows = [{"anon_id": f"{i:03d}", "label": "NEGATIVE" if i % 5 else "POSITIVE", "has_images": i % 2 == 0}
                for i in order]
        return pd.DataFrame(rows)

    def test_the_negatives_are_always_the_same_and_come_from_the_image_sample(self):
        chosen = fresh_ocr.negatives_subset(self.people(range(100)), 10, seed=42)
        self.assertEqual(len(chosen), 10)
        self.assertTrue(all(int(a) % 2 == 0 and int(a) % 5 for a in chosen))  # selected for images, never a positive
        self.assertEqual(fresh_ocr.negatives_subset(self.people(reversed(range(100))), 10, seed=42), chosen)

    def test_asking_for_more_than_there_are_takes_them_all(self):
        self.assertEqual(len(fresh_ocr.negatives_subset(self.people(range(100)), 1000, seed=42)), 40)


def session(account: str, profile: str = "paying"):
    fake = mock.Mock(profile_name=profile)
    fake.client.return_value.get_caller_identity.return_value = {"Account": account}
    return fake


def environment(**settings):
    """os.environ without any Rekognition setting of the real .env, plus the given ones."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("REKOGNITION_")}
    return mock.patch.dict(os.environ, {**env, **settings}, clear=True)


class AccountGuard(unittest.TestCase):
    def client(self, paying: str, corporate: str, named: str | None = None):
        """rekognition_client() when the profile's keys belong to `paying`, the default session
        to `corporate` and .env names the account `named` (by default, the profile's own)."""
        with environment(REKOGNITION_PROFILE="paying", REKOGNITION_ACCOUNT_ID=named or paying), \
                mock.patch.object(common, "rekognition_session", return_value=session(paying)), \
                mock.patch.object(common, "aws_session", return_value=session(corporate)), \
                mock.patch("builtins.print"):
            return common.rekognition_client()

    def test_the_named_account_is_accepted(self):
        self.assertIsNotNone(self.client("111111111111", "222222222222"))

    def test_the_corporate_account_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "same AWS account"):
            self.client("222222222222", "222222222222")

    def test_a_profile_of_an_account_that_was_not_named_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "not to the one named in REKOGNITION_ACCOUNT_ID"):
            self.client("333333333333", "222222222222", named="111111111111")

    def test_nothing_is_assumed_when_the_account_was_not_named(self):
        cases = (({}, "REKOGNITION_PROFILE and REKOGNITION_ACCOUNT_ID"),
                 ({"REKOGNITION_PROFILE": " ", "REKOGNITION_ACCOUNT_ID": "111111111111"}, "REKOGNITION_PROFILE missing"),
                 ({"REKOGNITION_PROFILE": "paying"}, "REKOGNITION_ACCOUNT_ID missing"),
                 ({"REKOGNITION_PROFILE": "paying", "REKOGNITION_ACCOUNT_ID": "my account"}, "12-digit"))
        for settings, message in cases:
            for entry in (common.paying_account, common.rekognition_session, common.rekognition_client):
                with self.subTest(settings=settings, entry=entry.__name__), environment(**settings), \
                        mock.patch("boto3.Session") as boto_session:
                    with self.assertRaisesRegex(RuntimeError, message):
                        entry()
                    boto_session.assert_not_called()  # no session at all, default or not

    def test_the_account_number_may_be_written_with_dashes(self):
        with environment(REKOGNITION_PROFILE="paying", REKOGNITION_ACCOUNT_ID="1111-1111-1111"):
            self.assertEqual(common.paying_account(), ("paying", "111111111111"))


if __name__ == "__main__":
    unittest.main()
