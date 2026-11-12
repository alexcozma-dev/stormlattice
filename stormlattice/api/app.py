"""Step 10: FastAPI backend. nginx on the VM proxies https://storm.alexcozma.dev to this app.

Run locally:  uvicorn stormlattice.api.app:app --reload
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI

from stormlattice import __version__
from stormlattice.config import mode

app = FastAPI(title="StormLattice", version=__version__)

DISCLAIMER = "Prototype for the Florida Poly AI Hackathon. Not for operational flight decisions."


@app.get("/health")
def health() -> dict:
    """Liveness check used by Docker and the deploy script."""
    return {
        "status": "ok",
        "version": __version__,
        "mode": mode(),
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/api/meta")
def meta() -> dict:
    """Banner info for the interface: data mode and the prototype disclaimer."""
    return {"mode": mode(), "disclaimer": DISCLAIMER}


# Planned endpoints:
#   GET  /api/forecast/latest      probability tiles + data timestamps + ensemble spread
#   POST /api/route                route risk, departure window, reroute
#   GET  /api/hazards              parsed NOTAM / SIGMET polygons
#   POST /api/briefing             LLM briefing for a route (rate-limited)
