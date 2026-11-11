"""Structural rule engine. Every rule returns PASS, FAIL or NOT_EVALUABLE.
Only FAIL should add risk. NOT_EVALUABLE is tracked as coverage.

A broken OCR read must end in NOT_EVALUABLE, never in FAIL: each rule only fails on evidence
that cannot be explained by a partial, reordered or one-character-off read.

Weights are NOT defined here: they come from the false-positive and detection rates measured
with scripts/12_ocr_coverage.py on the dataset.
"""
from dataclasses import dataclass
from datetime import date, timedelta

from .ocr_parse import MONTHS, BackFields, parse_front_numbers
from .registrars import registrars_in_office, term_ends

PASS, FAIL, NOT_EVALUABLE = "PASS", "FAIL", "NOT_EVALUABLE"


@dataclass
class RuleResult:
    rule: str
    status: str
    detail: str = ""


def _birth_and_issue_dates(back: BackFields) -> tuple[date | None, date | None]:
    """The yellow card prints exactly two dates: birth and issue. The earlier one is the birth
    date whatever order the OCR returned them in. Any other count is ambiguous -> (None, None)."""
    valid = sorted({d for d, _ in back.dates if d})
    return (valid[0], valid[1]) if len(valid) == 2 else (None, None)


def rule_code_format(back: BackFields) -> RuleResult:
    """The parser only accepts the 1-7-8-M/F-10-8 shape. Text that looks like a code but does not
    fit cannot be told apart from broken OCR -> NOT_EVALUABLE, never FAIL. In practice this rule
    measures how often the code is readable."""
    return RuleResult("R_CODE_FORMAT", PASS if back.code else NOT_EVALUABLE)


def rule_front_number_matches_code(front_text: str, back: BackFields) -> RuleResult:
    """The 10-digit number inside the back code must be the number printed on the front."""
    candidates = parse_front_numbers(front_text) if front_text else []
    if not back.code or not candidates:
        return RuleResult("R_NUMBER", NOT_EVALUABLE)
    code_number = back.code.number
    for c in candidates:
        if sum(a != b for a, b in zip(c.zfill(10), code_number)) <= 1:  # 1-char OCR tolerance
            return RuleResult("R_NUMBER", PASS)
    significant = code_number.lstrip("0")
    if any(c.lstrip("0") in significant for c in candidates):
        # e.g. the OCR split "1.012.345.678" and only "1.012.345" was read
        return RuleResult("R_NUMBER", NOT_EVALUABLE, "partial read of the front number")
    return RuleResult("R_NUMBER", FAIL, f"{len(candidates)} front candidate(s), none matches the code")


def rule_sex(back: BackFields) -> RuleResult:
    if not back.code or not back.printed_sex:
        return RuleResult("R_SEX", NOT_EVALUABLE)
    return RuleResult("R_SEX", PASS if back.code.sex == back.printed_sex else FAIL,
                      f"{back.code.sex} vs {back.printed_sex}")


def rule_production_vs_issue(back: BackFields) -> RuleResult:
    birth, issue = _birth_and_issue_dates(back)
    if not back.code or not back.code.production_date or not issue:
        return RuleResult("R_PRODUCTION", NOT_EVALUABLE)
    prod = back.code.production_date
    if prod < issue:
        return RuleResult("R_PRODUCTION", FAIL, f"production {prod} before issue {issue}")
    if birth and prod < birth + timedelta(days=18 * 365):
        return RuleResult("R_PRODUCTION", FAIL, f"production {prod} before age 18 (birth {birth})")
    return RuleResult("R_PRODUCTION", PASS)


def rule_registrar(back: BackFields) -> RuleResult:
    """The registrar printed under the signature must have been in office on the production date.
    Fallback with issue date only: fraud if the issue date is AFTER the signer's whole term."""
    if not back.registrar:
        return RuleResult("R_REGISTRAR", NOT_EVALUABLE, "no legible surname")
    prod = back.code.production_date if back.code and back.code.production_date else None
    if prod:
        expected = registrars_in_office(prod)
        return RuleResult("R_REGISTRAR", PASS if back.registrar in expected else FAIL,
                          f"{back.registrar} with production {prod}; expected {sorted(expected)}")
    _, issue = _birth_and_issue_dates(back)
    if not issue:
        return RuleResult("R_REGISTRAR", NOT_EVALUABLE, "no date")
    ends = term_ends(back.registrar)
    if ends and all(e is not None and issue > e for e in ends):
        return RuleResult("R_REGISTRAR", FAIL, f"issue {issue} after every term of {back.registrar}")
    return RuleResult("R_REGISTRAR", PASS, "issue date only")


def _month_kind(raw: str) -> str:
    """'valid', 'misread' (one letter away from a real month: an OCR slip) or 'malformed'."""
    if raw in MONTHS:
        return "valid"
    if len(raw) == 3 and any(sum(a != b for a, b in zip(raw, m)) == 1 for m in MONTHS):
        return "misread"
    return "malformed"


def rule_month_format(back: BackFields) -> RuleResult:
    """Months on the yellow card are ENE..DIC, three letters. Only the month token is judged
    (an impossible day number is an OCR problem, not this rule's). A month one letter away from
    a real one is treated as an OCR slip -> NOT_EVALUABLE."""
    if not back.dates:
        return RuleResult("R_MONTHS", NOT_EVALUABLE)
    kinds = {raw: _month_kind(raw) for _, raw in back.dates}
    malformed = sorted(raw for raw, kind in kinds.items() if kind == "malformed")
    if malformed:
        return RuleResult("R_MONTHS", FAIL, ",".join(malformed))
    misread = sorted(raw for raw, kind in kinds.items() if kind == "misread")
    if misread:
        return RuleResult("R_MONTHS", NOT_EVALUABLE, "possible OCR slip: " + ",".join(misread))
    return RuleResult("R_MONTHS", PASS)


def evaluate(front_text: str, back: BackFields) -> list[RuleResult]:
    return [
        rule_code_format(back),
        rule_front_number_matches_code(front_text, back),
        rule_sex(back),
        rule_production_vs_issue(back),
        rule_registrar(back),
        rule_month_format(back),
    ]
