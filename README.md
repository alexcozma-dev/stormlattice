# StormLattice

**Learned-rate stochastic nowcasting of Florida thunderstorms for Advanced Air Mobility.**

StormLattice predicts where Florida thunderstorms will form, move, grow and die over the next
60 minutes, and turns that forecast into a go / wait / reroute advisory for drone and air-taxi
routes. A human operator makes the final call.

Built for the Florida Poly High-Tech Showcase AI Hackathon 2026
("Cleared for Takeoff: AI for Florida's Skies").

> **Prototype.** Not for operational flight decisions. Forecasts may be wrong or stale.

Live demo: https://storm.alexcozma.dev

---

## How it works

| Step | Module | What it does | Kind |
| --- | --- | --- | --- |
| 1 | `stormlattice/data` | Pull radar (MRMS), lightning (GLM), winds (HRRR), geography layers | code |
| 2 | `stormlattice/cells` | Detect and track storm cells; label births and deaths | code |
| 3 | `stormlattice/motion` | Optical-flow motion field and advection baseline (pysteps) | physics |
| 4 | `stormlattice/features` | Sea-breeze geography and environment features | code |
| 5 | `stormlattice/models` | XGBoost birth/death rate models, calibrated | **ML** |
| 6 | `stormlattice/lattice` | Stochastic lattice: advect, die, cold pools, birth, trigger; 50-member ensemble | physics + ML |
| 7 | `stormlattice/risk` | Route risk from the ensemble, departure window, reroute | code |
| 8 | `stormlattice/llm` | Parse NOTAMs/SIGMETs/PIREPs to JSON; write a cited briefing | **LLM** |
| 9 | `tests/`, `scripts/evaluate.py` | CSI, Brier skill, reliability, ablations vs. baselines | validation |
| 10 | `stormlattice/api`, `web/` | FastAPI backend and map interface | app |

Full design write-up: see `docs/architecture.md`.

## Repository layout

```
config/          settings.yaml (grid, thresholds, lattice parameters)
data/            local data cache (not committed)
deploy/nginx/    reverse-proxy config for the VM
docker/          Dockerfile and compose file
docs/            design notes
notebooks/       exploration (not part of the app)
scripts/         one-off and batch jobs (fetch, train, evaluate)
stormlattice/    the Python package
tests/           pytest suite
web/             front-end
```

## Setup

Requires Python 3.10+.

```bash
git clone git@github.com:alexcozma-dev/stormlattice.git
cd stormlattice
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # app + pipeline
pip install -r requirements-train.txt    # training and notebooks (laptop only)
cp .env.example .env                     # then fill in keys
pytest
```

Run the API locally:

```bash
uvicorn stormlattice.api.app:app --reload
# open http://127.0.0.1:8000/health
```

## Data sources

All public and free:

- NOAA MRMS radar mosaic (AWS Open Data, `s3://noaa-mrms-pds`)
- GOES-East Geostationary Lightning Mapper (NOAA Open Data)
- HRRR model (NOAA Open Data)
- SEVIR storm event dataset (AWS Open Data, `s3://sevir`) for training
- NOAA shoreline, USGS hydrography and 3DEP elevation, NLCD land cover
- FAA NOTAM API; aviationweather.gov Data API (SIGMETs, PIREPs, METARs, TAFs)

## Responsible AI checklist

- [ ] Facts in briefings are grounded in cited sources, not model memory
- [ ] All calculations are done in code, never by the LLM
- [ ] A human makes the final go/no-go decision
- [ ] Missing or stale data leads to "advisory unavailable, treat as no-go"
- [ ] Uncertainty (ensemble spread) and data timestamps are visible
- [ ] Inputs are validated; prompt injection was tested
- [ ] No secrets in the repo; no personal data collected
- [ ] Simulated and replayed data are labeled
- [ ] A held-out test set was run and accuracy is reported below
- [ ] Prototype disclaimer is shown in the interface

## Results

_To be filled in from `scripts/evaluate.py` on the held-out test days._

## License

MIT. See `LICENSE`.
