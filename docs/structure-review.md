# Structure review: StormLattice

Reviewed Nov 15, 2026, against the "Current plan (Nov 15 revision)" in `docs/design.md`. No file was edited; this document is the only output. The science is unchanged throughout.

## Verdict

The style in `stormlattice/data/mrms.py` and `tests/test_mrms.py` is right and should be the template for everything else. The structure around it still describes the old plan: half the stubs are for cut work, the current plan's first-week modules have no home, and scripts cannot import the package.

About two hours of structural work fixes all of it. Do findings 1–4 in one sitting before writing new code, because the batch fetch due tonight is a script and needs a grid, an archive module and a place to write.

## Findings ranked by value

| # | Finding | Minutes |
| --- | --- | --- |
| 1 | Scripts cannot import `stormlattice`; the package is not installable | 5 |
| 2 | Stale stubs describe the cut design | 15 |
| 3 | The package mirrors the old ten steps and collides with other names | 20 |
| 4 | The current plan's modules and scripts have no home | 30 (stubs only) |
| 5 | No layout for training data, and a UTC date splits a storm evening in two | 15 |
| 6 | No layout for replay outputs; one global mode; map overlay rotation | 30 |
| 7 | The train/validation/test day list is not a tracked file | 10 |
| 8 | `settings.yaml` has cut keys and lacks the plan's keys; a test pins a tunable | 15 |
| 9 | `csi` lives in `scripts/` and a test imports it from there | 5 |
| 10 | README table, data sources and layout describe the old plan | 15 |
| 11 | `test_basics.py` is a grab-bag that breaks as one unit | 10 |
| 12 | Docker runs Python 3.12; the page cannot be viewed locally | 10 |
| 13 | Trained artifacts have no home or format | 5 (decision) |
| 14 | No `CLAUDE.md` | 10 (draft below) |
| 15 | Small items | 10 |

### 1. Scripts cannot import the package

- **Problem.** `python scripts/anything.py` from the repo root fails with `ModuleNotFoundError: No module named 'stormlattice'`, because Python puts `scripts/` on the path, not the root. Tests pass only because of `pythonpath = ["."]` in the pytest settings. `pip install -e .` fails as-is: "Multiple top-level packages discovered in a flat-layout". I confirmed both on a scratch copy.
- **Why it matters.** Every task in the next week is a script: batch fetch, headroom test, table build, training.
- **Smallest fix.** Add this to `pyproject.toml`, then run `pip install -e .` once. I tested this exact text.

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["stormlattice*"]
```

Keep the install editable. `config.py` finds `config/settings.yaml` relative to the source tree, which works for an editable install and in Docker, and would not for a regular install.

### 2. Stale stubs describe the cut design

- **Problem.** Several stubs document work the plan has cut or deferred.
- **Why it matters.** A stub's docstring reads as an instruction. An agent asked to "implement the lattice step" will build cold pools and κ triggers from `step.py`, and you will hand-type against the wrong spec.
- **Smallest fix.** Remove or trim as below.

| File | Action | Reason |
| --- | --- | --- |
| `stormlattice/cells/track.py` | Remove | Hungarian tracking is cut |
| `stormlattice/lattice/meanfield.py` | Remove | PDE solver is cut; the closed-form check replaces it |
| `stormlattice/data/environment.py` | Remove | GLM is cut, HRRR deferred; recreate as `hrrr.py` if the ablation happens |
| `scripts/fetch_mrms_frame.py` | Remove | Never implemented; the notebook does this; the archive script supersedes it |
| `stormlattice/lattice/step.py` | Rewrite the docstring; drop `POOL`, `count_fronts`, the κ and cold-pool fields | Lattice v1 has two states, object deaths, block births |
| `stormlattice/lattice/ensemble.py` | Fold `storm_probability` into the lattice module, then remove | Three real lines |
| `stormlattice/cells/detect.py` | Trim to a storm mask and connected components | VIL, echo tops and flashes are cut; objects at t0 are still needed |
| `stormlattice/features/geography.py` | Trim `FEATURE_NAMES` to coast distances, water fractions, land mask | NLCD, 3DEP and curvature are cut |
| `stormlattice/models/rates.py` | Drop `train_death_model` | The survival table replaces it |
| `stormlattice/llm/client.py`, `schemas.py` | Narrow to NOTAMs; drop the SIGMET and PIREP kinds | Those arrive decoded |
| `stormlattice/api/app.py` | Update the planned-endpoints comment | Reroute is cut |

### 3. The package mirrors the old ten steps

- **Problem.** Ten folders hold one or two files each, named after steps that no longer exist as numbered. Two names collide: `stormlattice/data/` (code) with `data/` (cache), and `stormlattice/models/` with wherever model files go.
- **Why it matters.** "Put it in data" is ambiguous for you and for an agent. Each new module also needs a "which folder" decision that adds nothing.
- **Smallest fix.** Flatten the science pipeline to one module per concept, with `git mv` so history follows. Leave `llm/` (three files that belong together) and `api/app.py` (named in the README, Dockerfile and nginx config) where they are. The rule from here on: a module becomes a folder only when it outgrows one file.

```
git mv stormlattice/data/mrms.py           stormlattice/mrms.py
git mv stormlattice/motion/advection.py    stormlattice/advection.py
git mv stormlattice/cells/detect.py        stormlattice/storms.py
git mv stormlattice/features/geography.py  stormlattice/geography.py
git mv stormlattice/models/rates.py        stormlattice/birth_model.py
git mv stormlattice/lattice/step.py        stormlattice/lattice.py
git mv stormlattice/risk/route.py          stormlattice/route.py
```

Then remove the emptied folders and their `__init__.py` files, and update five import lines in `tests/`.

### 4. The current plan's modules have no home

- **Problem.** Nothing in the code defines the 2 km grid or the 8 km blocks, and there is no place for the archive fetch, grid labels, survival table, feature table, baselines or metrics.
- **Why it matters.** These are the Nov 15–21 deliverables. `grid.py` comes first because the regrid, the blocks, the features and the map overlay all depend on it.
- **Smallest fix.** Create the modules below as docstring-only stubs, in the `mrms.py` style. Choose grid dimensions that are multiples of 4 so the 4×4 block pooling is exact.

Target layout (new files marked):

```
stormlattice/
  config.py        settings, env, data_dir, mode
  grid.py          NEW  the 2 km grid, lat/lon to grid index, block-maximum regrid, 4x4 block pooling
  mrms.py          bucket listing, download, load_florida (add: keys nearest each 10-minute mark)
  archive.py       NEW  fetch and crop one storm day to data/frames/<day>.npz; load_day
  advection.py     motion field, extrapolation, cumulative displacement
  storms.py        storm mask, connected components
  labels.py        NEW  birth labels per 8 km block per lead
  survival.py      NEW  build and read the survival table
  geography.py     static features from Natural Earth shapes
  features.py      NEW  radar-state and clock features; one table per storm day
  birth_model.py   train, calibrate, predict; probability and rate conversion
  baselines.py     NEW  persistence, probabilistic advection, block-by-hour climatology
  metrics.py       NEW  csi, frequency bias, Brier score and skill, bootstrap over days
  lattice.py       step, run, storm_probability, closed-form probability
  route.py         route risk, classify, departure window
  llm/             client, schemas, guardrails
  api/app.py
scripts/
  fetch_archive.py   NEW  loop over config/days.yaml; resumable
  check_headroom.py  NEW  early test 1
  check_buffer.py    NEW  early test 3
  check_labels.py    NEW  early test 5
  build_tables.py    NEW  labels and features per storm day
  train_birth.py     NEW
  evaluate.py        the evaluation table (calls baselines and metrics)
  build_replay.py    NEW  writes web/replay/<day>/
```

One convention keeps this readable: **package functions take plain arguments and return arrays; scripts read settings, loop over days, and do the file I/O and printing.** Anything that computes something goes in the package with a test. Nothing imports from `scripts/`.

### 5. Training data layout, and the storm day

- **Problem.** No layout exists for cropped data. Separately, S3 folders are UTC dates, and Florida's evening storms run past 00 UTC (8 pm EDT). A file per UTC date cuts one storm cycle in two, and adjacent halves can land in different splits.
- **Why it matters.** Splitting by day is the plan's main leakage rule. It only holds if a "day" contains a whole storm cycle.
- **Smallest fix.** Define a **storm day** as 10:00 UTC to 10:00 UTC the next day, named by its starting UTC date. All timestamps stay UTC. `archive.py` reads two S3 date folders per storm day. This regroups the same frames; it changes no label or feature.

```
data/                      gitignored, never committed
  grib/                    transient downloads, deleted after cropping
  frames/<YYYYMMDD>.npz    one storm day: dbz (t, ny, nx) float32, times_unix (t,) int64
  tables/<YYYYMMDD>.npz    feature rows and labels for that storm day
  live/                    on the VM: latest live outputs, same layout as a replay day
```

One file per storm day makes a split a list of file names and makes the batch fetch resumable (skip days whose file exists; write to `.part`, then rename, as `mrms.download` already does). Store times as Unix seconds because `.npz` cannot hold timezone-aware datetimes.

A full table of every block, issue time and lead runs to several million rows per day. Decide in `build_tables.py` how training days are thinned; validation and test days stay complete, as the plan requires.

### 6. Replay outputs

- **Problem.** Three gaps. There is no layout for precomputed replay outputs. `STORM_MODE` is one value per process, but the pitch opens in live mode and switches to replay. The grid is EPSG:5070, whose "up" is rotated about 7° from true north over Florida, so a PNG drawn from the grid and placed on a web map by its bounding box will be visibly misaligned.
- **Why it matters.** This is the Nov 22 deliverable and the demo.
- **Smallest fix.**
  - Write replay outputs as small PNG and JSON files to `web/replay/<YYYYMMDD>/` and commit them once the replay days are final. nginx already serves `web/`, so the VM gets them by `git pull` and the replay works with the API container down. If two days exceed about 20 MB, keep them in `data/replay/` and copy them to the VM instead.
  - Use the same file layout for `data/live/`. The page switches mode by loading a different folder, and the banner comes from that folder's `index.json`. `STORM_MODE` becomes the default shown on load.
  - Have `grid.py` export the grid's four corner coordinates in lon/lat and write them into `index.json`. A MapLibre image source takes four corners and handles the rotation. Leaflet's image overlay does not.

```
web/replay/<YYYYMMDD>/
  index.json              mode, issue times, leads, corner lon/lats, source ids
  obs_<HHMM>.png          observed reflectivity (UTC time)
  prob_<HHMM>_L<mm>.png   storm probability issued at HHMM for lead mm
  route_<HHMM>.json       route risk and departure window; the facts given to the briefing
  briefing_<HHMM>.json    cached briefing, so replay works with no LLM key
```

### 7. The day split is not a tracked file

- **Problem.** The plan says days are chosen by rule and replay days come from the test set "chosen before looking at results". Nothing records the choice.
- **Why it matters.** A committed list is the evidence, by commit date, that test days were fixed first. It is also the one place scripts and agents read splits from.
- **Smallest fix.** `config/days.yaml` with `train`, `validation`, `test` and `replay` lists of storm days. Commit it before training.

### 8. `settings.yaml` is out of date

- **Problem.** Cut keys remain: `cells.track_max_distance_km`, `cells.min_area_km2`, `lattice.cold_pool_*`, `lattice.kappa*`, `data.sevir_bucket`. `ensemble_members` is 50; the plan says 200. The plan's parameters are absent: cadence, storm-day start hour, block size, minimum pixels, persistence frames, the 20-minute quiet window, the advected-mask tolerance, the corridor endpoints and cruise speed.
- **Why it matters.** `test_settings_load` asserts `ensemble_members == 50`, so the planned change breaks a test for no reason.
- **Smallest fix.** Delete the cut keys, add the plan's, and change the test to check that keys exist and have sensible types. The YAML-to-dict approach itself is fine.

### 9. `csi` lives in `scripts/`

- **Problem.** `tests/test_basics.py` imports `from scripts.evaluate import csi`.
- **Why it matters.** It works only through the pytest path setting, and it makes a script importable code.
- **Smallest fix.** Move `csi` to `stormlattice/metrics.py`. After finding 1, delete `pythonpath = ["."]` so tests cannot import scripts by accident.

### 10. README

- **Problem.** The "How it works" table, the data sources and the layout list describe the old plan.
- **Why it matters.** The rules require clear documentation, and judges read the README.
- **Smallest fix.** Replace the table with the one below. In data sources, keep MRMS, FAA NOTAMs and aviationweather.gov; add Natural Earth; remove GLM, SEVIR, NLCD, USGS and 3DEP; mark HRRR as planned. Add `pip install -e .` to setup and `artifacts/` to the layout.

| Stage | Module | What it does | Kind |
| --- | --- | --- | --- |
| 1 | `mrms.py`, `archive.py`, `grid.py` | Fetch MRMS composite reflectivity, crop to Florida, regrid to a 2 km grid by block maximum, one file per storm day | code |
| 2 | `advection.py` | pysteps motion field; advected storm mask at each lead time (the classical baseline) | physics |
| 3 | `storms.py`, `labels.py`, `survival.py` | Storm objects; birth labels on 8 km blocks (storms that advection does not explain); survival table | code |
| 4 | `geography.py`, `features.py` | Coast distances, water fraction, clock time, lead time, radar state | code |
| 5 | `birth_model.py` | One calibrated XGBoost classifier: birth probability per block per lead | **ML** |
| 6 | `baselines.py`, `metrics.py`, `scripts/evaluate.py` | Brier skill and CSI against persistence, advection and climatology on held-out days | validation |
| 7 | `lattice.py` | 200-member ensemble: storms move and die as objects, births switch on blocks | physics + ML |
| 8 | `route.py` | Share of members in which the flight meets a storm; departure window | code |
| 9 | `llm/` | Parse free-text NOTAMs to validated JSON; write a cited briefing | **LLM** |
| 10 | `api/app.py`, `web/` | FastAPI backend and map page, replay and live | app |

### 11. Tests

- **Problem.** `test_basics.py` imports six modules, so one moved import fails the whole file at collection.
- **Why it matters.** Findings 2, 3 and 9 each move something it imports.
- **Smallest fix.** One `tests/test_<module>.py` per module, split as each module is touched. Keep helpers inline, as `FakeFS` is, until a second file needs them. Tests the plan already names or implies:

| File | Test |
| --- | --- |
| `test_grid.py` | a single 50 dBZ pixel survives the block-maximum regrid |
| `test_archive.py` | a storm day selects keys from two UTC folders (reuse `FakeFS`) |
| `test_labels.py` | new storm in an empty block is a birth; a storm under the advected mask is not; a one-frame flicker is not |
| `test_lattice.py` | ensemble mean matches the closed form (fixed seed, tolerance sized to the member count); an object dies whole |
| `test_route.py` | the existing `classify` tests; route risk on hand-made states |
| `test_metrics.py` | the existing `csi` test; Brier on a hand-computed case |

### 12. Docker and local serving

- **Problem.** The Dockerfile uses `python:3.12-slim`; the project targets 3.10 and the laptop runs 3.10.12. Locally, `uvicorn` serves only the API, so `web/index.html` cannot be opened against it.
- **Why it matters.** Two Python versions means two sets of wheels to debug. Without local serving, the page can only be tested on the VM.
- **Smallest fix.** Change the base image to `python:3.10-slim`. At the end of `app.py`, mount `web/` with FastAPI's `StaticFiles` (two lines), so local URLs match the VM's.

`storm.alexcozma.dev` returned 200 on `/` and 404 on `/health` on Nov 15, so the VM is still on the static-page config. Bring the container up well before Nov 22.

### 13. Trained artifacts

- **Problem.** No decision on where the model, calibration, survival table, climatology and geography arrays live or in what format.
- **Why it matters.** The VM needs them for live mode, and pickles break across library versions (requirements are unpinned).
- **Smallest fix.** A tracked `artifacts/` folder of small files. Save XGBoost with `save_model` to JSON. Save the isotonic calibration as its two threshold arrays in an `.npz` and apply it with `np.interp`. Add `COPY artifacts/ artifacts/` to the Dockerfile when the folder exists.

### 14. No `CLAUDE.md`

See the draft at the end.

### 15. Small items

- `docs/design.md:Zone.Identifier` is an untracked Windows download marker. Delete it and add `*:Zone.Identifier` to `.gitignore`.
- `mrms.py` imports `grib_full_defs_path` and never uses it. Delete it unless it is there on purpose.
- Module docstrings say "Step N" from the original design. Say what the module does instead.
- `requirements-train.txt`: `h5py` was for SEVIR and `rasterio` for NLCD and 3DEP. Remove them if nothing else needs them.
- `.env.example` names the old FAA API credentials. Rename when the NOTAM access arrives.
- `app.py` mentions a deploy script that does not exist. A five-line `deploy/deploy.sh` (pull, compose up, curl `/health`) is worth having before demo week.

## Fine as-is

- `mrms.py` and `test_mrms.py`: module docstring, small functions, comments that say why, a fake S3 in the test, opt-in network tests. Copy this style.
- `config.py`: YAML settings, `.env` secrets, `data_dir()`, `mode()`.
- `llm/guardrails.py`, `route.classify` and `departure_window`, the rate conversions in `rates.py`, and their tests.
- `llm/` as a folder and `api/app.py` at its current path.
- `docker/compose.yaml` and `deploy/nginx/storm.conf`. Two deployment folders is one more than needed and not worth moving.
- Two requirements files; unpinned versions are acceptable for three weeks.
- `data/.gitignore`, the `.gitignore`, `.env.example`, `LICENSE`.
- `notebooks/01_mrms_explore.py` as a `# %%` script.
- One design doc with a decision log; `docs/review-prompts.md`.
- Ruff at line length 110, target `py310`.

## Draft `CLAUDE.md`

```markdown
# StormLattice

Hackathon project: learned storm-birth probabilities plus a stochastic ensemble,
turned into a go / wait advisory for a Florida air-taxi route. Submission is due
Dec 2, 2026. Solo developer.

The plan is the "Current plan" section of docs/design.md. It is authoritative.
Steps 1-10 below it are the original design, kept for the explanations.

## How to work with me

I hand-type code to learn it. So:

- Explain the change first: what, why, and what I should understand afterwards.
- Propose small diffs in chat, about 30 lines at a time. Do not write or edit
  files unless I ask in that message.
- Never rewrite a whole file or refactor silently. One idea per change.
- Propose the test together with the function.
- If the plan seems wrong, say so. Do not work around it. Plan changes go in the
  decision log in docs/design.md.

## Commands

    pip install -r requirements-train.txt && pip install -e .
    pytest                              # offline tests
    STORM_NETWORK_TESTS=1 pytest -s     # also the live S3 tests
    ruff check .
    uvicorn stormlattice.api.app:app --reload

## Layout

- stormlattice/   one module per concept, in pipeline order: grid, mrms, archive,
                  advection, storms, labels, survival, geography, features,
                  birth_model, baselines, metrics, lattice, route; llm/; api/
- scripts/        command-line jobs. They read settings, loop over days, do file
                  I/O and print. Nothing imports from scripts/.
- tests/          tests/test_<module>.py for each module
- config/         settings.yaml (parameters), days.yaml (train/validation/test/replay)
- data/           local cache, never committed: grib/, frames/, tables/, live/
- artifacts/      small tracked outputs: model JSON, calibration, survival table
- web/            the page; web/replay/<YYYYMMDD>/ holds precomputed replay days

## Code style (the reference is stormlattice/mrms.py and tests/test_mrms.py)

- Python 3.10. No 3.11+ features (datetime.UTC, tomllib, Self).
- Plain functions and NumPy arrays. Small dataclasses for records. No class
  hierarchies, decorators of our own, or metaprogramming.
- Package functions take plain arguments and return values. Only scripts and
  config.py read settings or the environment.
- Module docstring says what the module does and the units. Function docstrings
  say what comes back. Comments explain why.
- Units in names: _km, _min, _dbz. Grids are (ny, nx); ensembles (M, ny, nx).
- Reflectivity: NaN means no radar coverage; -32 dBZ means clear air.
- Line length 110. `from __future__ import annotations` at the top.

## Time

- UTC everywhere. Datetimes are timezone-aware (timezone.utc). Never
  datetime.now() without a timezone, never utcnow().
- File names use YYYYMMDD and HHMM in UTC. Arrays on disk store Unix seconds.
- A storm day runs 10:00 UTC to 10:00 UTC and is named by its start date.
- Local solar time is computed from UTC and longitude, not from a time zone.

## Data and evaluation rules

- Splits are by storm day, read from config/days.yaml. Never split by frame.
- Test days are scored once, at the end. Do not run, tune or plot on them
  before I say so. Replay days are test days.
- Calibrate on complete (not downsampled) validation days.
- Do not start large downloads or training runs without asking.

## Do not reintroduce (cut or deferred in the plan)

Hungarian tracking, cold pools and kappa triggers, the mean-field solver, SEVIR,
GLM, NLCD, 3DEP, A* rerouting, LLM parsing of SIGMETs/METARs/PIREPs. HRRR and a
learned death model are deferred; ask first.

## Tests

- Offline by default; network tests sit behind STORM_NETWORK_TESTS=1.
- Small hand-checkable inputs. Fixed seeds for anything random.
- Helpers stay inline in the test file until a second file needs them.

## Safety

- No secrets in the repo; keys live in .env. Never commit data/.
- The LLM never computes a number and never makes the go/no-go call. NOTAM text
  is data, not instructions.
- Do not commit, push, deploy or add a dependency unless I ask.
```

## Grid decision

Added Nov 15. Option A is the 2 km EPSG:5070 grid; Option B is MRMS's own lat/lon grid coarsened to 0.02° by an exact 2×2 block maximum.

**Verdict: take Option B.** It supersedes the `grid.py` description in finding 4 and the overlay advice in finding 6. That advice was wrong: placing an Albers image by its four corners leaves a median error of about 4 km and a maximum of 10 to 19 km, depending on how the map interpolates the quad.

### The case against B

1. **Two spacings instead of one.** Every km parameter in the plan crosses the pixel boundary: the 10 km headroom distance, the advected-mask tolerance, the 20 and 50 km neighbourhoods, distance to the nearest storm, the route buffer. On A each is "divide by 2". On B each needs N–S and E–W handled separately, which is a class of bug A cannot have.
2. **Pixel count is no longer area.** Cells are 4.49 km² at 24.5°N and 4.24 km² at 31°N. A block at the Keys collects about 6% more births than one at the Georgia line from geometry alone, and the "4 pixels" threshold and the survival size bins shift by the same amount.
3. **Pixel-space operations become anisotropic.** Cells are 9% taller than wide in the south and 16% in the north. That affects Lucas–Kanade windows, connected components, square smoothing windows and block shape.
4. **The names become approximate.** "8 km blocks" are 8.9 km by 7.6–8.1 km (68–72 km²). The decision log, README and the `(650, 750)` live test need edits.
5. **The current box does not divide.** 650 × 750 source pixels is not a multiple of 8 in either direction, so the 0.08° blocks do not tile it.

### Why B still wins

The distortions are small against the plan's own tolerances, and A has distortions of its own.

| Check | Option A | Option B |
| --- | --- | --- |
| Source pixels per cell | 2 to 5 (64% get 4, 23% get 3, 8% get 2) | exactly 4 |
| Anisotropy | up to 2.8% at the south tip | 9–16%, known and fixed per row |
| Distance error with one constant spacing | up to 1.4% | up to 3.2% E–W at the box edges |
| Area per cell | constant | varies 6%, exact by row |
| Grid north against true north | rotated 5.1° to 9.7° across the box | aligned |
| Map overlay | full 2-D resample | 1-D row resample |
| Code | pyproj, a cached index map | reshape and max |

A 3% distance error on a 20 km radius is 0.6 km, under a third of a cell. The 6% area gradient is absorbed exactly by the block-by-hour climatology and is far below the factor-of-two width of any sensible size bin.

### What was checked

**pysteps (read in the installed 1.21.5 source):**

- `motion/lucaskanade.py` and the tracking and interpolation helpers it calls contain no pixel size, km or metadata reference. Windows and scales are in pixels and the motion field comes back in pixels per step.
- `extrapolation/semilagrangian.py` displaces pixel coordinates by a pixel velocity. Nothing metric.
- `noise/motion.py` takes one scalar `pixelsperkm`, so the motion-perturbation generator does assume square pixels. So do `kmperpixel` in the STEPS nowcast and the parametric FFT noise filter, which is isotropic in pixel space. The current plan uses none of these. If motion perturbation returns, pass the mean (about 0.48 pixels per km); the 7% directional error is small beside default parameters fitted on other radars.

I did not run pysteps on an anisotropic field.

**Labels, features, survival, lattice:** labels and rates are self-consistent because the label, the model and the lattice all use the same block. The lattice moves objects by a pixel displacement taken from a pixel motion field, so no km conversion is involved. Features and the survival table need the conditions below.

**Overlay (computed, not checked in the map libraries' source):** the 1-D claim is right. Web Mercator x is linear in longitude and y depends on latitude only. A lat/lon raster placed naively by its bounds is off by up to 5.4 km north–south near 27.8°N, which matches your figure. The simplest correct method: in `build_replay.py`, build output rows evenly spaced in Mercator y between the box edges and take the nearest source row for each. Then use an ordinary bounds overlay, `L.imageOverlay` in Leaflet or an image source with the four box corners in MapLibre. Both stretch linearly in Mercator, which is then exact to half a cell.

**HRRR (from memory, not checked):** either option needs a 2-D lookup from HRRR's Lambert grid. B is mildly favoured for winds. HRRR winds are grid-relative and must be rotated to east and north components once; on B those are the grid axes, while A needs a second rotation of 5° to 10° that varies across the box.

### Conditions for B

1. `grid.py` owns every conversion: `DY_KM` (2.216), `dx_km(lat)`, one mid-latitude `DX_KM` (1.97) for kernels, and cell area per row. No other module multiplies by 2 or calls a cosine.
2. Distances use `distance_transform_edt(mask, sampling=(DY_KM, DX_KM))`. Km-radius neighbourhoods use one elliptical footprint built from the same two numbers.
3. Wherever the plan says km² or "fraction of area" (headroom test, storm area within 20 and 50 km, survival size bins), sum cell areas by row, not pixel counts.
4. Choose a box whose spans are multiples of 0.08°. For example, `lat_min: 24.52` and `lon_min: -87.52` give 648 × 752 source pixels, 324 × 376 cells and 81 × 94 blocks.
5. Decide how the block maximum treats no-coverage pixels. A plain `max` lets one NaN blank the cell; `np.fmax.reduce` ignores NaN unless all four are NaN.
6. Say "0.02° cells, about 2 km" and "0.08° blocks, about 8 km". Add a decision-log row replacing the EPSG:5070 entry, and add condition 1 to `CLAUDE.md`.
