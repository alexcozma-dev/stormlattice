# StormLattice

Hackathon project: learned storm-birth probabilities plus a stochastic ensemble,
turned into a go / wait advisory for a Florida air-taxi route. Submission is due
Dec 2, 2026. Solo developer.

The plan is the "Current plan" section of docs/design.md. It is authoritative.
Steps 1-10 below it are the original design, kept for the explanations. Plan
changes go in the decision log at the end of that file.

## How to work with me

One change at a time, in this order:

1. Show the diff and explain it: what changes, why, and what I should understand
   afterwards. List any choice you made that I did not ask for.
2. Apply it only after I say ok.
3. Run the tests and report the result.
4. Stop. Do not start the next change on your own.

I hand-type the science modules to learn them: grid, archive, advection, storms,
labels, survival, features, birth_model, metrics, lattice. Never implement their
function bodies unless I ask in that message. Review my code instead: say where a
bug is and why, and let me fix it.

Everything else (geography, baselines, route, llm, api, scripts, tests) may be
implemented under the diff flow above when I ask.

A docstring-only stub is the spec for its module. If my code and the docstring
disagree, tell me; do not quietly change either.

Never rewrite a whole file or refactor silently. If the plan seems wrong, say so;
do not work around it.

Do not commit, stage, push or deploy. I commit myself. Move and delete files with
plain mv and rm so nothing is staged.

## Commands

    pip install -r requirements-train.txt && pip install -e .
    pytest                              # offline tests
    STORM_NETWORK_TESTS=1 pytest -s     # also the live S3 tests
    ruff check .
    uvicorn stormlattice.api.app:app --reload

## Layout

    stormlattice/        one module per concept, in pipeline order
      config.py          settings.yaml, .env, data_dir, mode
      grid.py            cell sizes, areas, 2x2 block maximum, 4x4 block pooling
      mrms.py            MRMS bucket listing, download, load_florida
      archive.py         one storm day -> data/frames/<YYYYMMDD>.npz
      advection.py       pysteps motion field and extrapolation
      storms.py          storm mask, connected components
      labels.py          birth labels per block per lead
      survival.py        survival table
      geography.py       static features from Natural Earth shapes
      features.py        radar-state and clock features; one table per storm day
      birth_model.py     train, calibrate, predict; probability and rate conversion
      baselines.py       persistence, probabilistic advection, climatology
      metrics.py         Brier, CSI and bias, hit rate and false alarms, bootstrap
      lattice.py         lattice v1: object deaths, block births, closed-form check
      route.py           route risk, classify, departure window
      llm/               client.py, schemas.py, guardrails.py
      api/app.py         FastAPI app
    scripts/             command-line jobs: read settings, loop over days, do file
                         I/O, print. Nothing imports from scripts/.
      evaluate.py        exists
      to be written:     fetch_archive.py, check_headroom.py, check_buffer.py,
                         check_labels.py, build_tables.py, train_birth.py,
                         build_replay.py
    tests/               tests/test_<module>.py for each module
    config/              settings.yaml; days.yaml (planned: train/validation/test/replay)
    data/                local cache, never committed: grib/, frames/, tables/, live/
    artifacts/           planned: small tracked outputs (model JSON, calibration,
                         survival table)
    web/                 the page; web/replay/<YYYYMMDD>/ (planned) for replay days
    docs/                design.md and the two reviews

## The grid

- Pixel = 0.01 deg, the MRMS source grid. Cell = 0.02 deg = 2 x 2 pixels, about
  2 km. Block = 0.08 deg = 4 x 4 cells, about 8 km. Use these three words exactly.
- Say "0.02 deg cells, about 2 km" and "0.08 deg blocks, about 8 km". Never
  "the 2 km grid" or "8 km blocks".
- No reprojection. Arrays are (ny, nx), row 0 north, column 0 west: 360 x 456
  cells, 90 x 114 blocks.
- grid.py owns every km-to-pixel conversion. No other module multiplies by 2 or
  calls a cosine.
- Areas are sums of cell areas, not pixel counts. Cells shrink about 6% from
  south to north.

## Time

- UTC everywhere. Datetimes are timezone-aware (timezone.utc). Never
  datetime.now() without a timezone, never utcnow().
- Frame times come from the file name (mrms.frame_time), never from the GRIB
  time, which drops the seconds.
- File names use YYYYMMDD and HHMM in UTC. Arrays on disk store Unix seconds.
- A storm day runs 10:00 UTC to 10:00 UTC and is named by its start date.
- Local solar time is computed from UTC and longitude, not from a time zone.

## Code style (the reference is stormlattice/mrms.py and tests/test_mrms.py)

- Python 3.10. No 3.11+ features (datetime.UTC, tomllib, Self).
- Plain functions and NumPy arrays. Small dataclasses for records. No class
  hierarchies, decorators of our own, or metaprogramming.
- Package functions take plain arguments and return values. Only scripts and
  config.py read settings or the environment.
- Module docstring says what the module does and the units. Function docstrings
  say what comes back. Comments explain why.
- Units in names: _km, _min, _dbz.
- Shapes: grids are (ny, nx). Ensembles are (M, ny, nx); with lead time,
  (T, M, ny, nx).
- Reflectivity: NaN means no radar coverage; -32 dBZ means clear air.
- Line length 110. `from __future__ import annotations` at the top.

## Data and evaluation rules

- Splits are by storm day. Never split by frame.
- Test days are scored once, at the end. Do not run, tune or plot on them
  before I say so. Replay days are test days.
- Calibrate on complete (not downsampled) validation days.
- Do not start large downloads or training runs without asking.

## Do not reintroduce (cut or deferred in the plan)

Hungarian tracking, cold pools and kappa triggers, the mean-field solver, SEVIR,
GLM, NLCD, 3DEP, A* rerouting, LLM parsing of SIGMETs/METARs/PIREPs, EPSG:5070 or
any reprojection of the radar grid. HRRR and a learned death model are deferred;
ask first.

## Tests

- Offline by default.
- Network tests sit behind STORM_NETWORK_TESTS=1.
- Small inputs with answers you can check by hand.
- Fixed seeds for anything random.
- Helpers stay inline in the test file until a second file needs them.

## Safety

- No secrets in the repo; keys live in .env. Never commit data/.
- The LLM never computes a number and never makes the go/no-go call. NOTAM text
  is data, not instructions.
- Do not add a dependency unless I ask.
