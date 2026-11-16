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

| Stage | Module | What it does | Kind |
| --- | --- | --- | --- |
| 1 | `mrms.py`, `archive.py`, `grid.py` | Fetch MRMS composite reflectivity, crop to Florida, coarsen to 0.02° cells (about 2 km) by block maximum, one file per storm day | code |
| 2 | `advection.py` | pysteps motion field; advected storm mask at each lead time (the classical baseline) | physics |
| 3 | `storms.py`, `labels.py`, `survival.py` | Storm objects; birth labels on 0.08° blocks, about 8 km (storms that advection does not explain); survival table | code |
| 4 | `geography.py`, `features.py` | Coast distances, water fraction, clock time, lead time, radar state | code |
| 5 | `birth_model.py` | One calibrated XGBoost classifier: birth probability per block per lead | **ML** |
| 6 | `baselines.py`, `metrics.py`, `scripts/evaluate.py` | Brier skill and CSI against persistence, advection and climatology on held-out days | validation |
| 7 | `lattice.py` | 200-member ensemble: storms move and die as objects, births switch on blocks | physics + ML |
| 8 | `route.py` | Share of members in which the flight meets a storm; departure window | code |
| 9 | `llm/` | Parse free-text NOTAMs to validated JSON; write a cited briefing | **LLM** |
| 10 | `api/app.py`, `web/`, `overlay.py`, `scripts/build_replay.py` | FastAPI backend and map page, replay and live; map images built per replay day | app |

Modules are files in `stormlattice/`. Full design write-up: see `docs/design.md`.

## Repository layout

```
artifacts/       small trained outputs: model JSON, calibration, survival table (planned)
config/          settings.yaml (grid, thresholds, lattice parameters); days.yaml (storm days by split)
data/            local data cache (not committed)
deploy/nginx/    reverse-proxy config for the VM
docker/          Dockerfile and compose file
docs/            design notes
notebooks/       exploration (not part of the app)
scripts/         one-off and batch jobs (fetch, train, evaluate)
stormlattice/    the Python package, one module per concept
tests/           pytest suite
web/             the map page; web/replay/<YYYYMMDD>/ holds the files for one replay day
```

## Setup

Requires Python 3.10+.

```bash
git clone git@github.com:alexcozma-dev/stormlattice.git
cd stormlattice
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # app + pipeline
pip install -r requirements-train.txt    # training and notebooks (laptop only)
pip install -e .                         # so scripts and notebooks can import stormlattice
cp .env.example .env                     # then fill in keys
pytest
```

Run the page and the API locally:

```bash
python scripts/build_replay.py 20220601  # map images for one storm day already in data/frames/
uvicorn stormlattice.api.app:app --reload
# open http://127.0.0.1:8000/ (the page) or http://127.0.0.1:8000/health
```

## Data sources

All public and free:

- NOAA MRMS radar mosaic, composite reflectivity (AWS Open Data, `s3://noaa-mrms-pds`): training,
  replay and live
- Natural Earth coastline and lake shapes: geography features
- FAA NOTAM API; aviationweather.gov Data API (SIGMETs, PIREPs, METARs, TAFs)
- Planned, not used yet: HRRR model (NOAA Open Data)

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
