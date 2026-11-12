"""History of the team's ID verifier, saved under <output>/private/verifier_history/casos/
(one JSON per case, with real document numbers: local only).

Each case has the ID number (`cc`), a verdict (`real`: FALSA, AUTÉNTICA or Pendiente) and who
set it (`real_origen`). The tool marks a case as forged by itself when its own score is 61 or
more ("Claude"); only a verdict set by an analyst is a confirmation. Older cases do not record
who set the verdict.
"""
import json
from pathlib import Path

FORGED_BY_ANALYST = "FORGED, set by an analyst"
FORGED_UNKNOWN_ORIGIN = "FORGED, origin not recorded"
FORGED_BY_TOOL = "FORGED, marked by the tool from its score"
GENUINE_BY_ANALYST = "GENUINE, set by an analyst"


def history_dir(out: Path) -> Path:
    return out / "private" / "verifier_history" / "casos"


def groups(out: Path) -> dict[str, set]:
    """Document numbers (digits, no leading zeros) of the saved history, by verdict and by who
    set it. A number confirmed by an analyst is not counted again in a weaker group. Every set
    is empty when no history has been saved."""
    found = {name: set() for name in (FORGED_BY_ANALYST, FORGED_UNKNOWN_ORIGIN, FORGED_BY_TOOL, GENUINE_BY_ANALYST)}
    folder = history_dir(out)
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        case = json.loads(path.read_text(encoding="utf-8"))
        number = "".join(ch for ch in str(case.get("cc") or "") if ch.isdigit()).lstrip("0")
        verdict, origin = case.get("real"), case.get("real_origen")
        if not number:
            continue
        if verdict == "FALSA":
            found[FORGED_BY_ANALYST if origin == "Analista" else
                  FORGED_BY_TOOL if origin == "Claude" else FORGED_UNKNOWN_ORIGIN].add(number)
        elif verdict == "AUTÉNTICA" and origin == "Analista":
            found[GENUINE_BY_ANALYST].add(number)
    found[FORGED_UNKNOWN_ORIGIN] -= found[FORGED_BY_ANALYST]
    found[FORGED_BY_TOOL] -= found[FORGED_BY_ANALYST] | found[FORGED_UNKNOWN_ORIGIN]
    return found


def forged_by_analyst(out: Path) -> set:
    """Document numbers an analyst confirmed as forged: the only ones used as a label."""
    return groups(out)[FORGED_BY_ANALYST]
