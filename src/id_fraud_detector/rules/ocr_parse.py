"""Field extraction from the Rekognition `detect_text` output already stored in
CustomerIdValidationResult.data_front_id / data_back_id.

Principle: when a field cannot be read in its expected shape, return None ("not evaluable").
A broken OCR read is never interpreted as a malformed document field.
"""
import re
from dataclasses import dataclass, field
from datetime import date

MONTHS = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AGO": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12}

# Typical OCR confusions inside numeric groups. All matching runs on upper-cased text, so a
# lowercase "l" read instead of "1" arrives here as "L". Only applied to text already located
# as a numeric field, never to the whole OCR text.
_NUM_CONFUSIONS = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1",
                                 "S": "5", "B": "8", "Z": "2", "G": "6"})
_NUMLIKE = "[0-9OQDILSBZG|]"

MIN_CONFIDENCE = 80.0  # minimum Rekognition confidence for a line to be used


def ocr_lines(payload: dict | None, min_confidence: float = MIN_CONFIDENCE) -> list[str]:
    """LINE detections above the confidence threshold, in reading order.

    Accepts the stored column value, {"detected_text": {"TextDetections": [...]}, ...} (the
    shape mikro_back writes), or a bare detect_text response, {"TextDetections": [...]}.
    """
    if not isinstance(payload, dict):
        return []
    if "TextDetections" not in payload:
        payload = payload.get("detected_text")
        if not isinstance(payload, dict):
            return []
    return [str(d.get("DetectedText") or "") for d in payload.get("TextDetections") or []
            if isinstance(d, dict) and d.get("Type") == "LINE" and (d.get("Confidence") or 0) >= min_confidence]


def _normalize_digits(s: str) -> str:
    return s.translate(_NUM_CONFUSIONS)


def _true_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


@dataclass
class BackCode:
    """The machine code printed on the back: LETTER-PLACE(7)-SEQ(8)-SEX-NUMBER(10)-YYYYMMDD."""
    letter: str
    place: str
    sequence: str
    sex: str
    number: str
    production_date: date | None
    raw: str


_RE_CODE = re.compile(
    r"([PAR])\s*[-–]?\s*(\d{7})\s*[-–]?\s*(\d{8})\s*[-–]?\s*([MF])\s*[-–]?\s*(\d{10})\s*[-–]?\s*(\d{8})")
_RE_CODE_CANDIDATE = re.compile(
    rf"[PAR]\s*[-–]?\s*{_NUMLIKE}{{7}}[\s\-–]*{_NUMLIKE}{{8}}[\s\-–]*[MF][\s\-–]*{_NUMLIKE}{{10}}[\s\-–]*{_NUMLIKE}{{8}}")
_RE_SERIAL = re.compile(rf"({_NUMLIKE}{{10}})\s*([A-Z])\s*({_NUMLIKE}{{1,2}})(?![0-9A-Z])")
_RE_DATE = re.compile(r"\b(\d{1,2})\s*[-–]\s*([A-Z]{3,4})\s*[-–]?\s*(\d{4})\b")
_RE_FRONT_NUMBER = re.compile(
    rf"(?<![0-9A-Z.])(?:{_NUMLIKE}{{1,3}}(?:\.{_NUMLIKE}{{3}}){{1,3}}|{_NUMLIKE}{{6,10}})(?![0-9A-Z])")


def _yyyymmdd(s: str) -> date | None:
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def parse_back_code(back_text: str) -> BackCode | None:
    """Find the back code in the concatenated OCR text. None when nothing fits the shape."""
    for candidate in _RE_CODE_CANDIDATE.findall(back_text.upper()):
        m = _RE_CODE.search(_normalize_digits(candidate))
        if m:
            letter, place, seq, sex, number, ymd = m.groups()
            return BackCode(letter, place, seq, sex, number, _yyyymmdd(ymd), candidate)
    return None


def parse_serial(back_text: str) -> tuple[str, str, str] | None:
    """(10 digits, letter, suffix) of the back serial, or None.

    OCR confusions are fixed only inside the two numeric groups, so the serial letter is kept
    as read (a letter such as B or S must not be turned into a digit).
    """
    for digits, letter, suffix in _RE_SERIAL.findall(back_text.upper()):
        if _true_digits(digits) < 7:  # mostly letters: a word, not a serial
            continue
        return _normalize_digits(digits), letter, _normalize_digits(suffix)
    return None


def parse_printed_dates(text: str) -> list[tuple[date | None, str]]:
    """Printed 'DD-MMM-YYYY' dates. Returns (date or None if it cannot be built, raw month)."""
    out = []
    for d, mon, y in _RE_DATE.findall(text.upper()):
        n = MONTHS.get(mon) if len(mon) == 3 else None  # 4-letter months only exist on the digital card
        try:
            out.append((date(int(y), n, int(d)) if n else None, mon))
        except (ValueError, TypeError):
            out.append((None, mon))
    return out


def parse_front_numbers(front_text: str) -> list[str]:
    """Every plausible ID number printed on the front, digits only and unpadded, longest first.

    A number is "plausible" when it has 6 to 10 digits, with or without thousands dots. An
    undotted 8-digit value that reads as a valid YYYYMMDD date is skipped; a dotted one never
    is, because IDs in the 19 and 20 million ranges are real and dates are not printed with dots.
    """
    out: list[str] = []
    for raw in _RE_FRONT_NUMBER.findall(front_text.upper()):
        compact = raw.replace(".", "")
        if _true_digits(compact) * 2 < len(compact):  # mostly letters: a word, not a number
            continue
        n = _normalize_digits(compact)
        if not 6 <= len(n) <= 10:
            continue
        if "." not in raw and len(n) == 8 and 1900 <= int(n[:4]) <= 2100 and _yyyymmdd(n):
            continue
        if n not in out:
            out.append(n)
    return sorted(out, key=len, reverse=True)


def parse_front_number(front_text: str) -> str | None:
    """Longest plausible ID number on the front, zero-padded to 10 digits (coverage reporting)."""
    numbers = parse_front_numbers(front_text)
    return numbers[0].zfill(10) if numbers else None


@dataclass
class BackFields:
    code: BackCode | None
    serial: tuple | None
    dates: list = field(default_factory=list)   # [(date | None, raw_month)]
    registrar: str | None = None
    printed_sex: str | None = None
    text: str = ""


def extract_back_fields(payload: dict | None) -> BackFields:
    """Parse the back side. `payload` is the stored data_back_id value or a bare detect_text."""
    from .registrars import detect_registrar

    text = " ".join(ocr_lines(payload))
    upper = text.upper()
    m = re.search(r"\bSEXO\b[^MF]{0,6}\b([MF])\b", upper)
    return BackFields(
        code=parse_back_code(text),
        serial=parse_serial(text),
        dates=parse_printed_dates(text),
        registrar=detect_registrar(upper),
        printed_sex=m.group(1) if m else None,
        text=text,
    )


def front_text(payload: dict | None) -> str:
    """Concatenated front-side text. `payload` is the stored data_front_id value or a bare detect_text."""
    return " ".join(ocr_lines(payload))
