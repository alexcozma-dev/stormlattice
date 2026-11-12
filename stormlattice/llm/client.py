"""Step 8: LLM calls. Provider and model come from .env (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY).

Two jobs only:
    parse_products(texts) -> ParsedProducts   (schema-constrained JSON; text is untrusted data)
    write_briefing(facts) -> str              (cites source ids; checked by guardrails)
The LLM never computes risk and never makes the go/no-go decision.
"""

from __future__ import annotations

from .schemas import ParsedProducts


def parse_products(texts: dict[str, str]) -> ParsedProducts:
    """Parse NOTAM / SIGMET / PIREP text (keyed by source id) into validated hazards."""
    raise NotImplementedError


def write_briefing(facts: dict) -> str:
    """Write a short cited briefing from structured facts computed in code."""
    raise NotImplementedError
