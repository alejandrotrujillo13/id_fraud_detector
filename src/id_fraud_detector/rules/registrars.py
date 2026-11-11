"""National Registrars (Registraduría Nacional del Estado Civil) and their terms.

Source: table provided by the verification team, cross-checked with press coverage.
Only terms since 1999 matter: IDs issued before the yellow hologram card lost validity in
2010, so no valid card should carry an earlier signature.
"""
import re
from datetime import date

# (surname key to look for in OCR text, full name, term start, term end). end=None -> in office.
REGISTRARS = [
    ("DUQUE", "Iván Duque Escobar", date(1999, 12, 6), date(2002, 8, 24)),
    ("RENGIFO", "Alma Beatriz Rengifo López", date(2002, 8, 25), date(2006, 12, 31)),
    ("GALINDO", "Juan Carlos Galindo Vácha (1st term)", date(2007, 1, 1), date(2007, 12, 5)),
    ("SANCHEZ", "Carlos Ariel Sánchez Torres", date(2007, 12, 6), date(2015, 12, 3)),
    ("GALINDO", "Juan Carlos Galindo Vácha (2nd term)", date(2015, 12, 4), date(2019, 12, 5)),
    ("VEGA", "Alexander Vega Rocha", date(2019, 12, 6), date(2023, 12, 5)),
    ("PENAGOS", "Hernán Penagos Giraldo", date(2023, 12, 6), None),
]

# Grey zones: date ranges where neither signature is penalized because the exact handover day
# is unconfirmed (team spreadsheet: 21-Nov-2023; press: 6-Dec-2023).
GREY_ZONES = [
    (date(2023, 11, 21), date(2023, 12, 6), {"VEGA", "PENAGOS"}),
]

# OCR variants of the surname printed under the signature. Matched as whole words.
# A bare "VEGA" is not accepted: it is also a place name printed on the back (birth or issue
# place "LA VEGA", "VEGACHI"), so that registrar needs a second token of the name.
ALIASES = {
    "DUQUE": ["DUQUE"],
    "RENGIFO": ["RENGIFO", "RENGIF0"],
    "GALINDO": ["GALINDO", "GALIND0", "VACHA", "VÁCHA"],
    "SANCHEZ": ["SANCHEZ", "SÁNCHEZ", "SANCHES"],
    "VEGA": ["VEGA ROCHA", "VEGA R0CHA", "ALEXANDER VEGA", "ROCHA", "R0CHA"],
    "PENAGOS": ["PENAGOS", "PENAG0S"],
}

_WORD = "A-ZÁÉÍÓÚÜÑ0-9"
_ALIAS_PATTERNS = {
    key: re.compile("|".join(rf"(?<![{_WORD}]){re.escape(a)}(?![{_WORD}])" for a in aliases))
    for key, aliases in ALIASES.items()
}


def registrars_in_office(day: date) -> set[str]:
    """Keys of the registrars who could have signed a card produced on `day`."""
    keys = {key for key, _, start, end in REGISTRARS if start <= day and (end is None or day <= end)}
    for start, end, extra in GREY_ZONES:
        if start <= day <= end:
            keys |= extra
    return keys


def term_ends(key: str) -> list[date | None]:
    return [end for k, _, _, end in REGISTRARS if k == key]


def detect_registrar(upper_text: str) -> str | None:
    """Registrar key whose surname appears in the (upper-cased) OCR text of the back side."""
    found = {key for key, pattern in _ALIAS_PATTERNS.items() if pattern.search(upper_text)}
    return found.pop() if len(found) == 1 else None  # none or ambiguous -> not evaluable
