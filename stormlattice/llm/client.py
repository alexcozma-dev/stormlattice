"""Step 8: LLM calls. Provider and model come from .env (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY).

Two jobs only:
    parse_notams(texts) -> ParsedNotams       (schema-constrained JSON; text is untrusted data)
    write_briefing(facts) -> str              (cites source ids; checked by guardrails)
SIGMETs, METARs and PIREPs arrive already decoded from aviationweather.gov, so they skip the LLM.
The LLM never computes risk and never makes the go/no-go decision.
"""

from __future__ import annotations

from .schemas import ParsedNotams


def parse_notams(texts: dict[str, str]) -> ParsedNotams:
    """Parse free-text NOTAMs (keyed by NOTAM number) into validated hazards."""
    raise NotImplementedError


def write_briefing(facts: dict) -> str:
    """Write a short cited briefing from structured facts computed in code."""
    raise NotImplementedError
