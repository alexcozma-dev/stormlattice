# Architecture

The full design write-up (physics and ML of every step, schedule, risks) lives in the
StormLattice design doc. This file tracks decisions made while building.

## Pipeline

radar + lightning -> cell tracking -> motion field
geography + HRRR -> XGBoost birth/death rates -> stochastic lattice (50 members) -> route risk -> operator
NOTAMs / SIGMETs / PIREPs -> LLM parser -> no-fly polygons -> route risk
route risk + parsed hazards + SHAP drivers -> LLM briefing -> operator

## Decision log

| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-11 | 2 km equal-area grid (EPSG:5070) over Florida | matches storm-cell scale; ~10^5 cells keeps the lattice fast in NumPy |
| 2026-11-11 | Lattice Design A (frozen rates + kappa triggers) for the MVP | fast and debuggable; Design B is research |
| 2026-11-11 | Host on one e2-small VM behind Cloudflare (Full strict) | fits the $50 credits; HTTPS on alexcozma.dev |
