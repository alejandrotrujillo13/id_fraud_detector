"""Rule engine tests on synthetic OCR (made-up numbers, no real documents).

Run:  python -m unittest discover tests
"""
import unittest

from id_fraud_detector.rules.engine import FAIL, NOT_EVALUABLE, PASS, evaluate
from id_fraud_detector.rules.ocr_parse import (extract_back_fields, ocr_lines, parse_back_code,
                                               parse_front_number, parse_front_numbers, parse_serial)
from id_fraud_detector.rules.registrars import detect_registrar

CODE = "A-1500150-00123456-M-1012345678-20100915"
FRONT = "REPUBLICA DE COLOMBIA NUMERO 1.012.345.678 APELLIDOS"
BACK = ["FECHA DE NACIMIENTO 15-MAR-1990", "SEXO M", "10-ABR-2008 BOGOTA D.C.",
        "CARLOS ARIEL SANCHEZ TORRES", CODE, "0012345678A 1"]


def detect_text(lines, confidence=99.0):
    return {"TextDetections": [{"Type": "LINE", "DetectedText": line, "Confidence": confidence} for line in lines]}


def stored(lines):
    """Shape written by mikro_back in data_front_id / data_back_id."""
    return {"detected_labels": {}, "detected_text": detect_text(lines), "detected_faces": {}}


def run(back_lines, front=FRONT):
    return {r.rule: r.status for r in evaluate(front, extract_back_fields(stored(back_lines)))}


class OcrShape(unittest.TestCase):
    def test_stored_shape_and_bare_shape_give_the_same_lines(self):
        self.assertEqual(ocr_lines(stored(BACK)), BACK)
        self.assertEqual(ocr_lines(detect_text(BACK)), BACK)

    def test_empty_or_unexpected_payloads_give_no_lines(self):
        for payload in (None, {}, {"detected_text": None}, {"detected_text": {}}, {"TextDetections": None}, "x"):
            self.assertEqual(ocr_lines(payload), [])

    def test_low_confidence_lines_are_dropped(self):
        self.assertEqual(ocr_lines(detect_text(BACK, confidence=50.0)), [])


class Parsers(unittest.TestCase):
    def test_back_code_survives_letter_for_digit_confusions(self):
        for noisy in ("A-l500150-00123456-M-1012345678-20100915", "A-15OO150-00123456-M-1012345678-20100915",
                      "A 1500150 00123456 M 1012345678 20100915"):
            code = parse_back_code(noisy)
            self.assertIsNotNone(code, noisy)
            self.assertEqual((code.place, code.sequence, code.sex, code.number), ("1500150", "00123456", "M", "1012345678"))

    def test_serial_keeps_its_letter(self):
        self.assertEqual(parse_serial("0012345678A 1"), ("0012345678", "A", "1"))
        self.assertEqual(parse_serial("0012345678B 1"), ("0012345678", "B", "1"))
        self.assertEqual(parse_serial("OO12345678S 12"), ("0012345678", "S", "12"))
        self.assertIsNone(parse_serial("BOGOTA D.C. COLOMBIA"))

    def test_front_number(self):
        self.assertEqual(parse_front_number(FRONT), "1012345678")
        self.assertEqual(parse_front_number("NUMERO 19.123.456"), "0019123456")   # 19-million range is a real ID
        self.assertEqual(parse_front_number("NUMERO 79.123.456"), "0079123456")
        self.assertIsNone(parse_front_number("20100915"))                          # undotted valid date
        self.assertIsNone(parse_front_number("REPUBLICA DE COLOMBIA"))
        self.assertEqual(parse_front_numbers("NUMERO 1.012.345. 678"), ["1012345"])

    def test_registrar(self):
        self.assertEqual(detect_registrar("CARLOS ARIEL SANCHEZ TORRES"), "SANCHEZ")
        self.assertEqual(detect_registrar("ALEXANDER VEGA ROCHA"), "VEGA")
        self.assertIsNone(detect_registrar("15-MAR-1990 LA VEGA (CUNDINAMARCA)"))
        self.assertIsNone(detect_registrar("VEGACHI (ANTIOQUIA)"))
        self.assertEqual(detect_registrar("LA VEGA (CUNDINAMARCA) CARLOS ARIEL SANCHEZ TORRES"), "SANCHEZ")
        self.assertIsNone(detect_registrar("SANCHEZ PENAGOS"))  # ambiguous


class Rules(unittest.TestCase):
    def test_genuine_card_passes_everything(self):
        self.assertEqual(set(run(BACK).values()), {PASS})

    def test_nothing_readable_is_not_evaluable(self):
        self.assertEqual(set(run([]).values()), {NOT_EVALUABLE})

    # --- broken OCR must never end in FAIL -------------------------------------------------
    def test_dates_in_swapped_order(self):
        swapped = [BACK[2], BACK[0], BACK[1], BACK[3], BACK[4]]
        self.assertEqual(run(swapped)["R_PRODUCTION"], PASS)

    def test_month_one_letter_off(self):
        result = run(["15-MAH-1990", "10-ABR-2008", "SEXO M", BACK[3], CODE])
        self.assertEqual(result["R_MONTHS"], NOT_EVALUABLE)
        self.assertEqual(result["R_PRODUCTION"], NOT_EVALUABLE)

    def test_impossible_day_is_not_a_month_problem(self):
        self.assertEqual(run(["45-MAR-1990", "10-ABR-2008", CODE])["R_MONTHS"], PASS)

    def test_birthplace_la_vega_with_unreadable_registrar(self):
        result = run(["15-MAR-1990 LA VEGA (CUNDINAMARCA)", "10-ABR-2008", "SEXO M", CODE])
        self.assertEqual(result["R_REGISTRAR"], NOT_EVALUABLE)

    def test_front_number_split_by_ocr(self):
        self.assertEqual(run(BACK, front="NUMERO 1.012.345. 678")["R_NUMBER"], NOT_EVALUABLE)

    def test_front_number_one_digit_off(self):
        self.assertEqual(run(BACK, front="NUMERO 1.012.345.679")["R_NUMBER"], PASS)

    # --- real inconsistencies must still FAIL ----------------------------------------------
    def test_different_front_number(self):
        self.assertEqual(run(BACK, front="NUMERO 43.567.890")["R_NUMBER"], FAIL)

    def test_sex_mismatch(self):
        self.assertEqual(run([BACK[0], "SEXO F", BACK[2], BACK[3], CODE])["R_SEX"], FAIL)

    def test_production_before_issue(self):
        self.assertEqual(run(["15-MAR-1990", "10-ABR-2012", CODE])["R_PRODUCTION"], FAIL)

    def test_production_before_age_18(self):
        self.assertEqual(run(["15-MAR-2000", "10-ABR-2008", CODE])["R_PRODUCTION"], FAIL)

    def test_registrar_not_in_office_on_production_date(self):
        result = run([BACK[0], BACK[1], BACK[2], "ALEXANDER VEGA ROCHA", CODE])  # produced in 2010
        self.assertEqual(result["R_REGISTRAR"], FAIL)

    def test_registrar_grey_zone_2023(self):
        code = "A-1500150-00123456-M-1012345678-20231128"
        for name in ("ALEXANDER VEGA ROCHA", "HERNAN PENAGOS GIRALDO"):
            self.assertEqual(run(["15-MAR-1990", "10-ABR-2008", name, code])["R_REGISTRAR"], PASS, name)

    def test_malformed_months(self):
        self.assertEqual(run(["15-SEPT-1990", "10-ABR-2008", CODE])["R_MONTHS"], FAIL)
        self.assertEqual(run(["15-XYZ-1990", "10-ABR-2008", CODE])["R_MONTHS"], FAIL)


if __name__ == "__main__":
    unittest.main()
