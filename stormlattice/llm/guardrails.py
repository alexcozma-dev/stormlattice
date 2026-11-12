"""Step 8 guardrails: the briefing may not contain numbers that aren't in its input facts."""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Every number in `text`, normalized (no trailing zeros, no thousands separators)."""
    found = set()
    for match in _NUMBER.findall(text.replace(",", "")):
        value = float(match)
        found.add(f"{value:g}")
    return found


def ungrounded_numbers(briefing: str, facts: str) -> set[str]:
    """Numbers that appear in the briefing but not in the facts it was given. Empty set = pass."""
    return numbers_in(briefing) - numbers_in(facts)
