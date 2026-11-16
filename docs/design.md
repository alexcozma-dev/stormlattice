# StormLattice: Learned-Rate Stochastic Nowcasting for Florida Air Mobility

Design doc, Nov 11, 2026; revised Nov 15 after an external architecture review (docs/architecture-review.md). Author: Alex Cozma.

**How to read this doc:** the *Current plan* section is authoritative. Steps 1–10 are the original design, kept for the physics and ML explanations; each starts with a note saying what changed. The decision log at the end records every change and why.

## Overview

StormLattice predicts where Florida thunderstorms will form, move, grow and die over the next 60 minutes, then turns that into a go/no-go risk for a drone or air-taxi route. A human operator makes the final call.

Moving existing storms forward is mostly a solved problem. Predicting where *new* storms will pop up is the hard part, and in Florida those births are driven by geography: sea breezes from both coasts collide inland, lakes make their own breezes, and outflow from old storms triggers new ones. StormLattice treats storms like reacting species on a map. Machine learning learns the local birth and death rates, and a stochastic lattice model (the storm version of a spatial Gillespie simulation) rolls those rates forward into an ensemble of possible futures.

**The pipeline at a glance**

1. **Data:** live radar, lightning and winds, plus fixed geography layers.
2. **Cells:** find and track individual storm cells, and label which ones are newborn.
3. **Motion:** estimate how the storm field is moving (the classical baseline).
4. **Geography:** turn coastlines, lakes and land cover into features.
5. **Learned rates:** XGBoost predicts each grid cell's chance of a storm being born, and each storm's chance of dying.
6. **Stochastic lattice:** advect, kill, birth and trigger storms on a grid, 50 times over, to get a probability map.
7. **Route risk:** convert the ensemble into the chance a given flight meets a storm, and suggest a reroute or a later departure.
8. **LLM layer:** parse NOTAMs and aviation weather text into structured data, and write a cited briefing.
9. **Validation:** score everything on held-out storm days against the classical baseline.
10. **Interface:** a live map with the ensemble animation, the route card and the briefing, hosted on Google Cloud.

Architecture (from the original diagram): radar + lightning -> cell tracking -> motion field; geography + HRRR winds -> XGBoost birth/death rates -> stochastic lattice (50-member ensemble) -> route risk (wait or reroute) -> operator decides. NOTAMs / SIGMETs / PIREPs / METARs -> LLM parser (text to hazard JSON) -> no-fly zones -> route risk. Route risk + parsed hazards -> LLM briefing (cited, no numbers of its own) -> operator. Only the XGBoost models and the two LLM steps are AI, and each feeds a step that can't run without it.

**Where the AI is essential:** without the learned rates, the system can only drag today's storms along. It can't see the afternoon sea-breeze storms coming. Without the LLM, NOTAM and SIGMET hazards never reach the router.


## Current plan (Nov 15 revision)

The core claim is narrow and testable: **on held-out Florida summer days, adding learned storm-birth probabilities to advection beats advection alone at 30–60 minute lead times, and the gain comes from births.** Everything is built to prove that first; the lattice and the app go on top.

### The minimum version (about five working days)

1. **Radar archive.** MRMS composite reflectivity for summer days chosen by a rule (e.g. every sixth day, June–August, keeping quiet days), at 10-minute cadence. Crop to Florida and coarsen MRMS's 0.01° grid to **0.02° cells (about 2 km)** by an exact 2×2 **block maximum** (averaging dilutes cores below 35 dBZ). Store only the cropped arrays as compressed `.npz`; delete each GRIB after cropping.
2. **Advection.** pysteps motion and extrapolation, giving the advected storm mask at each lead time.
3. **Grid labels (no tracking).** A block is a **birth at lead L** if it holds ≥35 dBZ (at least 4 cells, persisting two frames) in the window [t0+L, t0+L+10], held none in the previous 20 minutes, and is not covered by the advected mask (allow a small tolerance region so motion errors don't become fake births). Blocks are **0.08°, about 8 km** (4×4 cells), and the label, the predicted rate and the simulated footprint all use the same block. Births are labelled and scored only inside the **evaluation region**: the domain minus a 72 km margin on every side, so storms drifting in from outside the box are never counted as births.
4. **Features at t0.** Distance to each coast, water fraction and land mask (from Natural Earth shapes), local solar time, day of year, **lead time L**, and the radar state: neighbourhood maximum dBZ and its 10-minute trend (sub-35 dBZ echoes are the strongest cheap predictor), distance to the nearest storm, storm area within 20 and 50 km.
5. **Model.** One XGBoost classifier with shallow trees and a large `min_child_weight` (the real sample size is the number of days), calibrated with isotonic regression on **undownsampled** validation days. Output: birth probability per block per lead.
6. **Score.** Against persistence, advection, a probabilistic advection baseline and climatology (block × hour), at 30, 45 and 60 minutes on held-out days (see Evaluation below).

### The lattice, version 1 (on top of the minimum version)

- Two states only (empty, active), Design A, κ = 0. No cold-pool state yet.
- Existing storms are **objects** (connected components at t0). Each moves by its **cumulative displacement** from t0 (iterated nearest-neighbour advection stalls in weak flow) and **dies as a whole** with a hazard from an empirical **survival table** (fraction of the advected footprint still ≥35 dBZ, by storm size and lead).
- Births switch on a **whole 8 km block** with the learned rate for that lead.
- **200 members** (50 gives ±4-point noise at the green/amber threshold).
- Unit test: with these rules the ensemble mean has a closed form from the cumulative hazards; the simulation must match it. This replaces the SSA patch test and the mean-field PDE solver.

### LLM role

Free-text NOTAM parsing (schema-validated, injection-tested) and the cited briefing. SIGMETs, METARs and PIREPs come already decoded from aviationweather.gov, so the LLM doesn't parse those. NOTAM API access was requested from the FAA NOTAM Service Center on Nov 15; fallback is saved NOTAM text labeled as sample data.

### Evaluation (replaces the Step 9 table)

| Forecast | What it shows |
| --- | --- |
| Persistence | the floor |
| Advection (deterministic) | the classical method |
| Advection, probabilistic (neighbourhood-smoothed, threshold tuned on validation days) | a fair baseline for a probabilistic forecast |
| Advection + deaths only (survival table) | isolates the effect of decay |
| Advection + climatological birth rates | isolates "having births" from "learning births" |
| Advection + learned birth rates | the claim |
| Lattice v1 (route risk only) | coherent futures for the route card |
| Direct neighbourhood XGBoost ("any storm within d km at t0+L") | honest benchmark for what the lattice adds; fallback if it slips |

Lead with **Brier skill score** against block × hour climatology; report CSI with frequency bias, birth hit rate with false-alarm ratio, and a bootstrap interval over test days plus a win count ("better on 11 of 14 days"). Demo replay days come from the test set, chosen before looking at results.

### Early tests (run first)

1. **Headroom test:** on 3–5 summer afternoons, the fraction of observed ≥35 dBZ area at +30/+45/+60 min lying more than 10 km from the advected t0 storms. This is the ceiling of the whole claim.
2. **Fetch timing:** fetch and crop 3 days, time it, extrapolate; start the full batch overnight.
3. **Buffer test:** route risk from *observed* radar on a replay afternoon for 5, 10 and 20 km buffers (the SunTrax–MCO flight is ~15 minutes).
4. **Climatology vs. first XGBoost** on 10 train / 3 validation days (Brier).
5. **Label sanity:** one day of births on the map; births per day; share lasting one frame.
6. Later: e2-small cycle in Docker (`--cpus=0.5 --memory=2g`); the model on a clear December frame; demo with Wi-Fi off and no LLM key.

### Schedule

| Dates | Deliverable | Decision point |
| --- | --- | --- |
| Nov 15–17 | Batch fetch running overnight; regrid; advection baseline; headroom test. FAA request sent; teammate confirmed by Nov 20. | Headroom number known |
| Nov 18–21 | Grid labels, feature table, climatology baseline, first XGBoost, first proof chart on validation days | Does XGBoost beat climatology? Sets the pitch |
| Nov 22 | Thin page on the VM: observed-radar replay, route card with advection risk, LLM briefing from those facts | Both hard rules (LLM, GCP) satisfied |
| Nov 23–27 | Lattice v1, route risk, departure window, NOTAM parsing. Thanksgiving Nov 26 | |
| Nov 28–Dec 1 | Test set run once, accuracy panel, video recorded Nov 30, code frozen Dec 1 | |

### Cut or deferred

| Item | Status | Why |
| --- | --- | --- |
| SEVIR | Cut | different algorithm and era than live MRMS; few Florida tiles |
| GOES GLM lightning | Cut | thousands of files per day; MRMS has lightning density grids if needed |
| NLCD, 3DEP, coastline curvature | Cut | Natural Earth coast and water are enough; elevation is noise in Florida |
| Hungarian tracking, split/merge logic | Cut | replaced by grid labels |
| Intensity regression model | Cut | already optional |
| Exact SSA patch test, mean-field PDE solver | Cut | replaced by the closed-form check |
| A* reroute, energy gauge | Cut | short corridor; the departure window carries the value |
| deck.gl 3D columns | Cut | 2D map with overlays and a time slider |
| LLM parsing of SIGMETs/METARs/PIREPs | Cut | already decoded by the source API |
| Spread–skill metric | Cut | reliability covers calibration |
| Cold-pool state, κ triggers, gust-front layer | Deferred to future work / research | nothing observes a cold pool to fit or validate it |
| HRRR features | Deferred, then added as an ablation | most valuable deferred item; use only forecasts published by t0 |
| XGBoost death model, VIL, echo tops | Deferred | survival table first |

### Pitch notes

Keep Gillespie, contact process and propensity out of the pitch ("we roll the dice 200 times"). Don't claim that forecasting storms from sea-breeze boundaries is new (NCAR's Auto-Nowcaster did that for Florida in the 1990s); the contribution is learned birth rates driving a stochastic ensemble for AAM route decisions.

## Step 1 · Data

> **Revised Nov 15:** MRMS composite reflectivity archive only (10-minute cadence, block-max regrid, cropped `.npz` storage). SEVIR, GLM, NLCD and 3DEP are cut; HRRR is deferred. Coast distances and water come from Natural Earth.

Everything runs on free public data: radar and lightning to see storms, model winds to move them, geography to explain where they start, and a curated storm dataset to train on.

| Layer | Source | Resolution | Used for |
| --- | --- | --- | --- |
| Radar mosaic (reflectivity, VIL, echo tops) | [NOAA MRMS on AWS](https://registry.opendata.aws/noaa-mrms-pds/), no account needed | about 1 km, updates every 2 min | live input, cell detection |
| Lightning flashes | GOES-East Geostationary Lightning Mapper (NOAA open data) | event points, about 20 s | cell features, aviation hazard |
| Winds, CAPE, moisture | HRRR forecast model (NOAA open data) | 3 km, hourly | steering flow, environment features |
| Coastline, lakes, land cover, elevation | NOAA shoreline, USGS hydrography, NLCD land cover, USGS 3DEP | 30 m to 1 km, resampled to the 0.02° grid | geography features |
| Training events | [SEVIR](https://arxiv.org/pdf/2207.07033) (radar VIL, satellite IR, lightning) | 384 km tiles, 4 h, 5-min frames | model training and testing |

### The physics: what radar actually measures

Weather radar sends out microwave pulses and listens for the echo from raindrops and hail. The echo strength is the reflectivity factor:

```latex
Z = \int N(D)\, D^{6}\, dD, \qquad \mathrm{dBZ} = 10 \log_{10} Z
```

N(D) is the number of drops of diameter D per unit volume. Because of the sixth power (Rayleigh scattering), a few big drops or hailstones dominate the signal. That's why about 35–40 dBZ is a common working threshold for a convective core: only strong updrafts hold drops that big aloft.

**VIL (vertically integrated liquid)** converts reflectivity at every height into liquid water and sums it up the column. A tall, wet column means a strong updraft, so VIL is a better storm-strength signal than surface reflectivity alone. It's also the field SEVIR uses for nowcasting benchmarks.

**Lightning** marks the mature stage of a storm, when ice and graupel collide in the updraft and separate charge. For aircraft, lightning is the hazard itself, so it's both an input feature and an output to warn about.

### Build notes

- Pull MRMS GRIB2 files with `s3fs` or the AWS CLI; decode with `xarray` + `cfgrib`.
- Crop everything to a Florida box (23.80–31.00° N, 87.52–78.40° W) and put it on one 0.02° lat/lon grid (360 × 456 cells, 90 × 114 blocks; see the Nov 15 grid decision and the Nov 16 box decision). One grid for every layer keeps the lattice and the features aligned.
- Filter SEVIR events to tiles that overlap Florida, then supplement with MRMS archives from two or three summers (June–September) for Florida-only training.
- Tag every dataset with its timestamp and whether it is live, historical replay or simulated. The interface shows all three labels.

## Step 2 · Storm-cell detection and tracking

> **Revised Nov 15:** frame-to-frame tracking and split/merge logic are cut. Births are labeled on 8 km blocks as storms advection does not explain (see Current plan). Connected components at t0 are still used, as the objects the lattice moves and kills. The 4 km² minimum area is about one 0.02° cell (4.2–4.5 km²), so require ≥4 pixels and two-frame persistence.

This step turns radar pixels into a list of storm *objects* with ages and histories. It also creates the training labels: which cells were just born, and which just died.

### The physics: a storm cell's life cycle

A single thunderstorm cell lives roughly 30–60 minutes and passes through three stages:

1. **Cumulus (growing):** a warm, moist parcel rises because it is lighter than the air around it. Its buoyancy comes from CAPE (convective available potential energy). Only an updraft exists, and radar echoes appear and grow.
2. **Mature:** rain and hail get heavy enough to fall, dragging air down and cooling it by evaporation. Updraft and downdraft coexist. Reflectivity, VIL and lightning peak here.
3. **Dissipating:** the downdraft wins and cuts off the warm inflow. The cell rains itself out, and its cold outflow spreads across the ground (the *cold pool*, Step 6).

So a cell has a natural birth, peak and death, which is exactly what the lattice in Step 6 models.

### The algorithm

1. **Threshold:** mark pixels with composite reflectivity at or above 35 dBZ (tune between 35 and 40).
2. **Connected components:** group touching marked pixels into objects with `scipy.ndimage.label`. Drop objects smaller than about 4 km² (noise, ground clutter).
3. **Features per cell:** area, centroid, max dBZ, max and mean VIL, echo-top height, lightning flashes inside it over the last 5 minutes, and the trends of each over the last 10–15 minutes.
4. **Track between frames:** move each cell from the previous frame forward by the motion field (Step 3), then match predicted and observed cells with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on a centroid-distance and overlap cost. A match carries the cell's ID and age forward.
5. **Label events:**
   - **Birth:** a cell with no match in the previous frame that isn't a split from a neighbor.
   - **Death:** a tracked cell with no match in the next frame that didn't merge into a neighbor.
   - **Splits and merges:** flagged separately so they don't count as fake births or deaths.

### Where the ML comes in

No learning happens here, but every label the models train on comes from this step. Tracking bugs become label noise. Validate it early by plotting tracks on a few storm days and checking that ages and lifetimes look like 30–60 minute cells, not flickering one-frame blobs.

This object-and-track approach is how operational systems such as [MeteoSwiss COALITION-3](https://www.meteoswiss.admin.ch/about-us/research-and-cooperation/projects/2020/coalition-3.html) build their thunderstorm "cell biographies" before applying XGBoost.

## Step 3 · Motion field and the advection baseline

> **Revised Nov 15:** unchanged, plus a probabilistic version of this baseline for fair scoring (see Evaluation). Lattice objects move by cumulative displacement from t0, not step by step.

This step estimates how the whole storm field is moving and builds the classical forecast that the AI has to beat. It assumes storms move but never grow, decay or form.

### The physics: steering and propagation

A storm's motion has two parts:

- **Advection:** the cell is carried by the mean wind through the depth of the cloud, the *steering flow* (roughly the 850–300 hPa average wind).
- **Propagation:** new cells keep forming on one side of a storm, usually where its cold outflow meets the incoming warm air, so the storm *system* appears to drift in a direction the wind alone wouldn't predict.

The baseline captures only advection. Propagation is exactly what the learned birth rates and the lattice add.

### The math: Lagrangian persistence

The baseline assumes each rain feature is a conserved tracer moved by a velocity field (u, v):

```latex
\frac{\partial R}{\partial t} + u\,\frac{\partial R}{\partial x} + v\,\frac{\partial R}{\partial y} = 0 \quad\Longleftrightarrow\quad R(\mathbf{x}, t+\tau) = R(\mathbf{x} - \boldsymbol{\alpha}(\tau), t)
```

This is the pure advection equation, solved along characteristics. **Optical flow** estimates (u, v) by finding the displacement field that best maps one radar frame onto the next. The Lucas–Kanade method does this by least squares over small windows, assuming brightness is conserved and motion is locally smooth.

### Build notes

- Use [pysteps](https://gmd.copernicus.org/articles/12/4185/2019/): `pysteps.motion.get_method("LK")` for the motion field and `pysteps.extrapolation.semilagrangian` to push the latest frame forward 10–60 minutes.
- Smooth the optical-flow field and blend it toward the HRRR steering wind at longer lead times, where radar-only motion gets noisy.
- This forecast is **Baseline 1** in Step 9. Persistence ("nothing moves") is Baseline 0.
- The motion field is reused three times: for tracking in Step 2, as a feature in Step 5, and for advection inside the lattice in Step 6.

## Step 4 · Geography features

> **Revised Nov 15:** keep coast distances, water fraction, land mask and solar time from Natural Earth; cut curvature, NLCD and elevation; HRRR deferred. Add the radar-state features (neighbourhood max dBZ and trend, distance to nearest storm, nearby storm area).

This step encodes *where* in Florida storms like to be born. It's the storm version of the heterogeneous B(x) field from the author's Brusselator project, and builds on his NJ Highlands terrain pipeline.

### The physics: sea breezes and collisions

1. **Differential heating:** on a sunny morning, land heats much faster than water. Water has a higher heat capacity, mixes heat downward and loses some to evaporation.
2. **A pressure gradient forms:** the warm air column over land expands, so at a given height above the surface the pressure over land becomes higher than over the sea. Air flows seaward aloft, the surface pressure over land drops, and cool marine air flows *onshore* at the surface. That onshore flow is the sea breeze.
3. **The front moves inland:** the leading edge is a shallow density current. It pushes warm, moist air upward as it advances, often tens of kilometers inland by afternoon.
4. **Collision:** Florida is a narrow peninsula with breezes from both the Atlantic and the Gulf. Where they meet over the interior, the forced lift is strong. If the air is unstable (high CAPE), thunderstorms erupt along the collision line, typically mid to late afternoon in summer.
5. **Modifiers:** Lake Okeechobee makes its own lake breeze, cities form heat islands that strengthen updrafts, and the background wind direction decides which coast's breeze wins and how far it pushes inland.

### The features (per 0.02° grid cell)

| Feature | Why it matters |
| --- | --- |
| Distance to the Atlantic coast, distance to the Gulf coast | where each sea-breeze front tends to be at a given hour |
| Coastline curvature within 20 km | convex bulges in the coast focus breezes and their convergence |
| Water fraction within 5 and 20 km (lakes) | lake breezes, extra moisture |
| Land-cover fractions (urban, forest, wetland, crop) from NLCD | surface heating and moisture differ by land type |
| Elevation and local relief from 3DEP | small in Florida, but ridges add a little lift |
| Local solar time, day of year | sea breezes follow the sun |
| Onshore wind component on each coast (HRRR) | the regime: which breeze dominates and how far inland the collision sits |
| CAPE, precipitable water, low-level wind convergence (HRRR) | whether lift actually produces a storm |

The static layers are computed once and cached as arrays. The time-varying ones (HRRR, time of day) update every hour.

### Build notes

- Distance-to-coast fields: rasterize the shoreline, then `scipy.ndimage.distance_transform_edt`.
- Curvature: smooth the coastline and compute signed curvature, or take a simpler proxy, the land fraction within a 20 km disk.
- Keep the interaction between distance-to-coast and wind regime explicit. XGBoost can learn it, but a hand-made "distance to the expected collision zone" feature often helps a small dataset a lot.

## Step 5 · Learned rates with XGBoost

> **Revised Nov 15:** one birth model on 8 km blocks with **lead time as a feature** (target: birth in [t0+L, t0+L+10]), not a per-cell "within 4 km" target, which counted each birth ~13 times. The death model is replaced by an empirical survival table for now. Calibrate on undownsampled validation data. The probability-to-rate conversion below is unchanged.

This is the core AI component. Two gradient-boosted models turn radar, geography and environment features into calibrated probabilities: how likely a storm is to be **born** in each empty grid cell, and how likely each existing storm is to **die**, over the next time step (Δt = 10 min).

### The two models

| Model | Unit | Target | Key features |
| --- | --- | --- | --- |
| Birth | each empty grid cell, each time step | 1 if a tracked birth occurs within 4 km in the next 10 min | geography (Step 4), CAPE, moisture, convergence, distance to nearest storm, area of recent heavy rain upstream (a cold-pool proxy), time of day |
| Death | each tracked storm cell | 1 if the cell dies in the next 10 min | age, VIL and echo-top trends, lightning trend, size, CAPE around it, whether it sits inside its own outflow |

An optional third model regresses intensity change (next max VIL) so surviving storms can grow or weaken in the lattice.

### How gradient boosting works

A single decision tree splits the feature space into boxes ("distance to Gulf < 40 km and hour > 14") and predicts a constant in each box. One tree is crude. Boosting adds trees one at a time, each fitting the *errors* of the ensemble so far:

```latex
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\, h_m(\mathbf{x}), \qquad h_m \approx -\frac{\partial \mathcal{L}(y, F)}{\partial F}\bigg|_{F = F_{m-1}}
```

Each new tree approximates the negative gradient of the loss, so boosting is gradient descent in function space with learning rate η. For a yes/no target, F is a log-odds, the loss is log-loss, and the probability is p = 1 / (1 + e^(−F)). XGBoost adds a second-order (Newton) step and penalties on tree size and leaf weights to limit overfitting.

Why it fits this problem: the features are tabular and mixed (distances, fractions, trends, clock time), the interactions are nonlinear ("near the coast matters only when the wind is offshore"), it trains in minutes on a CPU, and it explains itself. Germany's weather service found tree methods trained faster and scored better in cross-validation than neural networks for storm-cell prediction ([DWD, EMS 2023](https://meetingorganizer.copernicus.org/EMS2023/EMS2023-451.html)).

### Three things that make or break it

1. **Split by day, not by frame.** Frames 5 minutes apart are nearly identical, so a random split leaks the answer into training. Hold out whole storm days (or whole seasons) for validation and testing.
2. **Handle the imbalance.** Births are rare: most empty cells stay empty. Downsample negatives or set `scale_pos_weight`, then fix the probabilities afterward (next point).
3. **Calibrate.** The lattice treats these numbers as physical rates, so a "10%" must mean 10% of the time. Fit isotonic regression on the validation days and check a reliability diagram (Step 9). Miscalibration compounds over 6 lattice steps.

### From probability to rate

The lattice can run at a different step than the model was trained on, so convert each probability to a rate, assuming births are Poisson within a step:

```latex
\lambda = -\frac{\ln(1 - p)}{\Delta t_{\text{train}}}, \qquad p_{\text{step}} = 1 - e^{-\lambda\, \Delta t_{\text{lattice}}}
```

This is the same propensity idea as Gillespie: λ is the per-site reaction propensity, learned from data rather than set by mass-action kinetics.

### Explanations for free

`shap` values give each prediction's top drivers ("+0.21 from sea-breeze collision zone, +0.12 from CAPE"). Those go to the LLM in Step 8, so the briefing can say *why* a storm is likely, grounded in the model rather than in the LLM's memory.

## Step 6 · The stochastic lattice

> **Revised Nov 15:** version 1 has two states (no cold pools), κ = 0, deaths drawn **per object** (per-pixel deaths erode storms into speckle that never clears a route), births **per 8 km block**, cumulative-displacement advection, and 200 members. The cold-pool physics below is kept as the research direction, not shipped.

This is the novel part. The learned rates from Step 5 drive a stochastic cellular model of Florida in which storms are advected by the wind, die, leave cold pools behind, and are born, either from the environment or where cold pools collide. Running it 50 times with different random draws gives a probability map of future storms, uncertainty included.

### The state

The grid is the 0.02° Florida grid from Step 1 (about 2 km), 360 × 456 ≈ 1.6 × 10⁵ cells. Each cell is in one of three states, an excitable-medium model in the family of the Greenberg–Hastings automaton:

| State | Meaning | Analog in excitable media (BZ reaction) |
| --- | --- | --- |
| 0 · empty | no storm, can host a birth | resting medium |
| 1 · active | a storm core (with an intensity and an age) | excited |
| 2 · cold pool | rain-cooled outflow, births suppressed | refractory |

The occupancy is 0 or 1 per cell, not a large molecule count, so this is a contact-process model rather than mass-action chemistry.

### The physics: cold pools

When a storm's downdraft hits the ground, evaporatively cooled air spreads out as a **gravity current**, a shallow, dense layer, typically about 1 km deep. Its front moves at roughly

```latex
c \approx \mathrm{Fr}\sqrt{g\, h\, \frac{\Delta\theta}{\theta}}
```

with Froude number Fr ≈ 0.7–1, depth h, and a temperature deficit Δθ of a few kelvin. For h = 1 km and Δθ = 3 K this gives about 7 m/s, so the front advances about 4 km (two grid cells) per 10 minutes.

Two effects matter for the model:

- **Suppression:** inside the pool the air is cool and stable, so new storms are unlikely. This is the refractory state.
- **Triggering:** at the front, warm air is lifted. Where two or more fronts collide, the lift is much stronger. Large-eddy simulations show that colliding fronts, rather than single ones, usually set off new updrafts ([Haerter et al.](https://arxiv.org/pdf/2002.03474)), and a simple model of circles that spawn new circles where three meet reproduces how storm clusters grow through the day ([Haerter et al. 2019](https://arxiv.org/abs/1810.05518)).

### The update rule (one step, Δt = 5 min)

1. **Advect:** shift active cells and cold pools by the motion field (semi-Lagrangian, as in Step 3).
2. **Die:** each active cell becomes a cold pool with probability p_death from the death model.
3. **Expand and recover:** each cold pool's radius grows at speed c; cold-pool cells return to empty with probability Δt / τ_cp. The lifetime τ_cp is a fitted parameter, probably on the order of an hour or two over land.
4. **Be born:** each empty cell becomes active with probability

```latex
p_{\text{birth}}(\mathbf{x}) = 1 - \exp\!\Big[-\big(\lambda_{\text{env}}(\mathbf{x}) + \kappa_2\, \mathbb{1}[n_{\text{fronts}} \ge 2] + \kappa_3\, \mathbb{1}[n_{\text{fronts}} \ge 3]\big)\,\Delta t\Big]
```

Here λ_env is the learned environmental rate from Step 5, n_fronts is the number of distinct cold-pool fronts within one cell of x, and κ₂ and κ₃ are fitted trigger strengths.

```python
# one lattice step for all M ensemble members at once; arrays are (M, ny, nx)
active, pool = advect(active, motion), advect(pool, motion)
dies   = active & (rng.random(active.shape) < p_death)
active, pool = active & ~dies, pool | dies
pool  &= rng.random(pool.shape) > dt / tau_cp
fronts = count_distinct_fronts(pool_labels)        # ndimage.label + dilation
lam    = lam_env + k2 * (fronts >= 2) + k3 * (fronts >= 3)
born   = ~active & ~pool & (rng.random(active.shape) < 1 - np.exp(-lam * dt))
active |= born
```

### Two designs

| Design | Rates | Pros | Cons |
| --- | --- | --- | --- |
| A · frozen rates (hackathon MVP) | XGBoost run once per forecast on environment-only features; triggering handled by the κ terms | fast (milliseconds a step in NumPy), easy to debug | κ must be fitted separately |
| B · learned propensity (research) | XGBoost re-queried every step on features recomputed from the *simulated* state | the ML learns triggering itself; a true learned propensity function | about 3×10⁷ predictions per forecast; needs batching or a coarser grid |

### Why a discrete-time update, not exact Gillespie

Exact Gillespie draws one event at a time from the total propensity. With 10⁵ sites and many events per minute, that's far too slow in Python. The per-cell Bernoulli update is the lattice analog of tau-leaping: it's accurate when each cell's per-step probability is small, with error that shrinks with Δt. Exact SSA on a small 32×32 patch is still a good check that the fast version gives the same statistics. This kind of birth–death lattice evolved by Gillespie has a precedent at climate scale in the stochastic multicloud model ([Khouider, Biello & Majda 2010](https://projecteuclid.org/euclid.cms/1266935019)).

### Ensembles

Run M = 50 members with different random seeds, and perturb the motion field slightly in each (pysteps has noise generators for this). For every cell and lead time:

- **Storm probability** = fraction of members in which the cell is active.
- **Spread** shows where the forecast is uncertain, which the interface displays (a Responsible AI checklist item).

### The deterministic limit (validation habit)

With triggering turned off, the ensemble mean must match the mean-field equations, a reaction–advection system for the storm fraction ρ and the cold-pool fraction q:

```latex
\partial_t \rho + \mathbf{u}\cdot\nabla\rho = \lambda(\mathbf{x},t)\,(1-\rho-q) - \mu\,\rho, \qquad \partial_t q + \mathbf{u}\cdot\nabla q = \mu\,\rho - q/\tau_{cp}
```

Solving this PDE and comparing it with the ensemble mean checks the stochastic code against its deterministic limit. With triggering on, the nonlinear κ terms turn the system into an excitable medium, and storm clusters can propagate like trigger waves.

## Step 7 · Route risk and the go/no-go call

> **Revised Nov 15:** the departure window is the main product; A* reroute and the energy gauge are cut. Choose the buffer with the buffer test on observed radar (the corridor flight is ~15 minutes).

This step answers the operator's actual question: if this aircraft leaves at this time on this route, how likely is it to meet a storm, and what's a safer option? The system advises, and a person decides.

### Risk along a route, done correctly

1. Sample the route every 1 km and compute the estimated arrival time at each point from the aircraft's speed.
2. For each ensemble member, the flight **fails** if any route point at its arrival time is within a buffer distance d of an active storm cell.
3. Route risk = fraction of the 50 members in which the flight fails.

This counts members, rather than multiplying per-point probabilities, because storm hits along a route are strongly correlated. One storm blocks many consecutive points, and multiplying as if they were independent badly overstates the risk.

### Settings the operator controls

- **Buffer d:** manned-aviation guidance traditionally recommends staying about 20 nautical miles from severe thunderstorms. Small drones and air taxis may set their own; keep it a visible, adjustable setting rather than hard-coding a number.
- **Thresholds:** for example green under 10%, amber 10–30%, red over 30%. These are placeholders, labeled as such, for the operator to set.

### Better options, computed in code

- **Departure window:** rerun the route check for departures every 10 minutes over the next hour and plot risk against departure time.
- **Reroute:** A* search on the grid with cost = flight time + w × storm probability at the arrival time, avoiding no-fly polygons from the LLM layer (Step 8).
- **Stretch:** an energy-reserve gauge. A reroute adds distance; a simple battery model shows whether the reserve still clears the minimum.

### Safe defaults (Responsible AI checklist)

- Radar older than 10 minutes, or any missing input: show **advisory unavailable, treat as no-go**, never a stale green.
- Every number in the interface carries its data timestamp.
- The operator must acknowledge the advisory; the system never dispatches a flight by itself.

## Step 8 · The LLM layer

> **Revised Nov 15:** the parser handles free-text NOTAMs only; SIGMETs, METARs and PIREPs arrive decoded from aviationweather.gov. Pass facts to the briefing as preformatted strings so the number check doesn't trip on rounding.

The LLM does two language jobs that code can't do well and the model can't do at all: it reads the aviation text products humans write, and it writes the briefing a human reads. It never computes a number or makes the call.

### Job 1: text in, structured hazards out

| Input | Source | What the LLM extracts |
| --- | --- | --- |
| NOTAMs (flight restrictions, closures, obstacles) | FAA NOTAM API | type, center and radius or polygon, altitude floor and ceiling, valid from and to |
| Convective SIGMETs and SIGMETs | aviationweather.gov Data API | hazard area polygon, tops, movement, valid time |
| Pilot reports (PIREPs) | aviationweather.gov Data API | location, altitude, turbulence or icing severity, time |
| METARs and TAFs at nearby airports | aviationweather.gov Data API | surface wind, gusts, thunderstorm codes, visibility |

The output is JSON that must pass a schema. Code then checks the geometry (valid polygon, inside Florida, sensible altitudes and times) and turns it into no-fly polygons for the router in Step 7. Anything that fails validation is shown to the operator as raw text, flagged "could not parse," and treated conservatively.

### Job 2: the cited briefing

The LLM receives only structured facts computed elsewhere: route risk, departure window, top SHAP drivers from Step 5, parsed hazards, and every source's ID and timestamp. It writes a short plain-language briefing in which each claim cites a source ID.

### Guardrails

- **No numbers of its own:** after generation, code checks that every number in the briefing appears in the input facts. Mismatch means the briefing is regenerated or replaced with a template.
- **Untrusted text:** NOTAM and PIREP text is treated as data, never as instructions. Test it with injected strings and confirm the schema-constrained output is unaffected.
- **No decisions:** the briefing describes; the operator decides.

### Model choice

Any API model with structured (JSON-schema) output works. Teams pay their own LLM costs, so use a small, cheap model for parsing and cache parsed NOTAMs, which change slowly. For development, a local model on the P16 laptop costs nothing.

## Step 9 · Validation

> **Revised Nov 15:** the baselines and metrics are replaced by the Evaluation section of the Current plan (adds deaths-only, climatological-births and probabilistic-advection rows; leads with Brier skill; drops spread–skill and standalone ROC AUC).

The claim to prove is narrow and testable: on held-out Florida storm days, StormLattice beats advection-only forecasting at 30–60 minute lead times, and the gain comes from predicting storm births. Every number goes in the README and on an "Accuracy" panel in the app.

### Data splits

Split by whole days or seasons, never by frames. For example: train on summers 2021–2023, validate (and calibrate) on 2024, test once on 2025. If you use SEVIR, split by event and keep events from the same day together.

### Baselines and ablations

| Forecast | What it shows |
| --- | --- |
| Persistence (nothing moves) | the floor |
| pysteps advection (Step 3) | the standard to beat |
| Climatology (birth frequency by location and hour) | whether the model knows more than "afternoons near the collision zone" |
| XGBoost without geography features | how much the sea-breeze features add (the research question) |
| Lattice without cold-pool triggering (κ = 0) | how much the excitable-medium physics adds |
| Full StormLattice | the result |

### Metrics

- **CSI (critical success index)** at 10, 20, 30, 45 and 60 minutes, thresholding the probability map. CSI = hits / (hits + misses + false alarms). Also report it pooled over 4×4 neighborhoods, since a storm forecast 2 km off is still useful.
- **Brier skill score** against climatology, for the probability maps.
- **Reliability diagram:** forecast probability vs. observed frequency.
- **ROC AUC** for the birth model, plus **birth hit rate**: the fraction of real births with forecast probability above a threshold within 10 km and 15 minutes.
- **Spread–skill:** whether ensemble spread is large where errors are large.

### What to expect, honestly

At 10–20 minutes, advection is very hard to beat, because storms simply haven't had time to change. The value should appear at 30–60 minutes and on afternoon sea-breeze days. If it doesn't, that's still a reportable result, and the ablation table says why.

## Step 10 · Interface, deployment and demo

> **Revised Nov 15:** a thin end-to-end page goes live by **Nov 22**; 2D map with overlays (no deck.gl columns); replay days come from the test set; precomputed replay needs no compute. Consider resizing to e2-medium after the Docker test.

The app is a single map page: storms erupt and drift across Florida, the operator draws a route, and a card answers go, wait or reroute, with a cited briefing beside it. Training happens on the laptop; the Google Cloud VM only runs the live pipeline and serves the page.

### The interface

- **Map:** MapLibre GL with deck.gl layers, for a tilted 3D view with animated storm columns. Leaflet is the fallback.
- **Layers:** observed radar, ensemble storm probability, cold pools, the sea-breeze "birth likelihood" field, and no-fly polygons from NOTAMs.
- **Time slider:** scrub from now to +60 minutes; a "play ensemble" button animates 10 of the 50 futures side by side.
- **Route card:** risk percentage with green, amber or red; the departure-window chart; a reroute button; the energy gauge as a stretch goal.
- **Briefing panel:** the Step 8 briefing with clickable source IDs.
- **Always visible:** data timestamps, a "LIVE / HISTORICAL REPLAY / SIMULATED" banner, the accuracy panel from Step 9, and a prototype disclaimer.

### The backend

- **FastAPI** (Python) serves forecasts as GeoJSON and PNG tiles.
- A **worker** pulls MRMS every 2 minutes and HRRR every hour, runs Steps 2–7, and caches the results; the page only reads the cache.
- Both run in **Docker** on one small **Compute Engine** VM (e2-small, 2 GB RAM). XGBoost inference and the NumPy lattice run on CPU.
- **Credits:** $50. Keep the disk small (25 GB), and keep training data on the laptop, not the cloud disk.
- API keys live in environment variables or Secret Manager, never in the public repo.

### Replay mode (the December problem)

December 3 is Florida's dry season, so a live demo may show no storms. Precompute two or three dramatic summer days, ideally a textbook sea-breeze collision day over the central peninsula, and replay them through the full live pipeline under a clear "HISTORICAL REPLAY" banner. Offer to switch to live data during Q&A.

### The 3–5 minute video

1. **Hook (20 s):** the 3D map at 1 pm on a summer day: clear skies, then storms erupt along a line down the middle of the state.
2. **Problem (30 s):** Florida leads the US in lightning; drones and air taxis fly low and slow, and pop-up storms are what grounds them.
3. **Walkthrough (2 min):** draw a route, see the risk, open the departure-window chart, wait, see green. Open the briefing and its citations.
4. **How the AI works (1 min):** learned birth rates from geography, the lattice ensemble, the LLM parsing NOTAMs.
5. **Proof (30 s):** the CSI chart against advection and the ablation result.
6. **Future work (20 s):** closed-loop learned propensities, energy-aware routing, noise footprints, launch restrictions.

All lead times and percentages in this script are placeholders until the test set is run.

## Schedule, risks and what comes after

The build runs Nov 12 to Dec 2. The rule is that something working and demoable exists at the end of every week, so a slip costs polish, not the project.

Planned timeline: data pipeline + tracking Nov 12–15; pysteps baseline Nov 14–16; geography features Nov 15–18; XGBoost rates + calibration Nov 17–21; stochastic lattice Nov 20–25; route risk + LLM layer Nov 23–27; interface + GCP deploy Nov 22–30; validation + replay days Nov 27–Dec 1; demo video Nov 30–Dec 2. Registration closes Nov 30; submission due Dec 2, 4 pm; live judging Dec 3. Thanksgiving (Nov 26) falls inside the interface and LLM work.

### Risks and fallbacks

| Risk | Fallback |
| --- | --- |
| No teammate by registration close (Nov 30); solo entries aren't allowed | post in #find-a-team; a teammate can own the interface or the video |
| Data downloads are slow or huge | crop to Florida at download time; start with 20–30 storm days, not whole seasons |
| Tracking produces noisy birth labels | raise the area and intensity thresholds; hand-check one day of tracks |
| Lattice is late or unstable | ship Design A only; triggering (κ) becomes future work |
| Model doesn't beat advection at short leads | lead with 30–60 minute results and initiation hit rate; report the honest short-lead result |
| Cloud credits run low | stop the VM when idle; serve precomputed replay days |
| No storms on Dec 3 | replay mode (Step 10) |

### Research after the hackathon

1. **Closed-loop learned propensity (Design B):** let XGBoost learn triggering from simulated cold-pool features, and compare it with the hand-built κ terms.
2. **Leakage length:** measure how far inland the coastline's effect on birth rates extends, by wind regime.
3. **Local initiation condition:** compare learned birth rates with a physics-based local instability criterion built from CAPE and convergence.
4. **Mean-field vs. lattice:** where does the excitable-medium behavior make the stochastic model beat its deterministic PDE limit?
5. **Write it up:** a student poster or paper with a faculty advisor as an independent study.

Before claiming novelty, do a proper literature search on convective-initiation nowcasting with geographic features and on stochastic lattice models of convection.

## Addendum (Nov 13): refinements after reading the federal AAM documents

These decisions were made after reading the DOT AAM National Strategy and Comprehensive Plan (Dec 2025) and FDOT/SunTrax material. They take priority over the original text where they differ.

- **Framing:** anchor the problem to DOT AAM National Strategy Recommendation 2.8 (enhanced weather detection, forecasting and reporting for AAM; AAM flies below 5,000 ft where weather is less served; move from nowcasting toward short-term prediction and decision-support tools).
- **Gust-front hazard layer** *(deferred Nov 15: nothing observes a cold pool, so it can't be validated; future work)*: show the lattice's cold-pool fronts as a "gust-front risk" layer and include them in route risk. AAM aircraft are sensitive to wind and gusts (battery limits, handling); gust fronts can arrive before the rain.
- **Demo corridor:** SunTrax Air (FDOT AAM test facility near Florida Poly) to Orlando International Airport, its vertiport partner. eVTOL-like profile (~150 mph cruise, under 5,000 ft). Replay summer sea-breeze days along I-4.
- **Validation framing:** present Step 9 as a "weather provider performance report" (the Comprehensive Plan calls for standards and certification of third-party weather data providers).
- **Machine-readable output:** document one JSON advisory format for the route/forecast API (future autonomous aircraft, 2035 target).
- **Stated limits:** storm-scale hazards on a ~2 km grid; not microscale (<2 km) wind. Future work: aircraft-reported weather feeding the rates.

## Decision log

What was decided while building, and why. Newest last; add a row for each change to the plan.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-11 | 2 km equal-area grid (EPSG:5070) over Florida (superseded Nov 15) | matches storm-cell scale; ~10^5 cells keeps the lattice fast in NumPy |
| 2026-11-11 | Lattice Design A (frozen rates + κ triggers) for the MVP | fast and debuggable; Design B is research |
| 2026-11-11 | Host on one e2-small VM behind Cloudflare (Full strict) | fits the $50 credits; HTTPS on alexcozma.dev |
| 2026-11-12 | MRMS flags: -999 -> NaN (no coverage), -99 -> -32 dBZ floor (clear air) | clear air is information, not missing data; ~86% of a typical frame is clear |
| 2026-11-13 | Frame the project on DOT AAM Strategy Rec. 2.8; SunTrax -> MCO demo corridor; gust-front layer | ties the work to a stated national need and a local, real AAM route |
| 2026-11-14 | Offline unit tests with a fake S3 + opt-in live tests (STORM_NETWORK_TESTS=1) | fixed inputs give known answers; live tests check real data still matches |
| 2026-11-14 | Stop tracking .idea/; keep all design notes in this one file | editor settings are machine-specific; one doc is easier to keep current |
| 2026-11-15 | Adopt the external architecture review: minimum version first, thin page live by Nov 22 | original scope was ~3x what 17 days allow; LLM and GCP rules were scheduled last |
| 2026-11-15 | Grid birth labels on 8 km blocks instead of Hungarian tracking | cheaper, less label noise; labels and metric agree by construction |
| 2026-11-15 | Lead time as a feature; one rate map per lead | rates trained for +10 min were reused out to +60 |
| 2026-11-15 | Lattice v1: two states, per-object deaths (survival table), per-block births, cumulative displacement, 200 members | per-pixel deaths and per-cell births had the wrong units; weak flow stalls step-wise advection |
| 2026-11-15 | Cold pools, κ triggers and the gust-front layer deferred to future work | no observation to fit or validate them |
| 2026-11-15 | Cut SEVIR, GLM, NLCD, 3DEP, curvature, A*, energy gauge, deck.gl, SIGMET/METAR parsing; defer HRRR | one data product (MRMS) for training, replay and live; LLM keeps the defensible jobs |
| 2026-11-15 | Evaluation adds deaths-only, climatological-births and probabilistic-advection rows; lead with Brier skill | without them, "the gain comes from births" was unsupported and CSI favoured any probabilistic forecast |
| 2026-11-15 | NOTAM API requested from the FAA NOTAM Service Center (7-AWA-NAIMES@faa.gov); fallback: saved NOTAM text labeled as sample | access is by request, not self-serve |
| 2026-11-15 | 0.02° lat/lon grid (MRMS 0.01° coarsened by exact 2×2 block max) replaces EPSG:5070; box 24.52–31.00° N, 87.52–80.00° W; grid.py owns every km conversion | every cell is the max of exactly 4 pixels; no reprojection; north-aligned; web overlay needs only a 1-D row warp. Cost: cells 2.22 km × 1.91–2.03 km, area varies 6% by row, handled in grid.py (docs/structure-review.md) |
| 2026-11-15 | Storm day = 10:00 to 10:00 UTC, named by its start date; one frames file per storm day | Florida's evening storms cross 00 UTC; splitting by UTC date would cut a storm cycle across splits |
| 2026-11-16 | Box widened to 23.80–31.00° N, 87.52–78.40° W (90 × 114 blocks, 360 × 456 cells); evaluation region = domain minus 72 km | the old east edge (80.00° W) sat on the Atlantic coast and the south edge on the Keys, so storms moving in from offshore would have been labelled births. 72 km = 15 m/s × 80 min (60 min lead + 20 min quiet window). −78.40 is near the limit of useful offshore radar coverage |
| 2026-11-16 | Days fixed by rule in config/days.yaml before any model: train Jun–Aug 2022–24 and validation Jun–Aug 2025 every 2nd day; test Jun–Aug 2026 every 3rd day; replay days are test days chosen by observed storms | split by whole summers so no weather system spans two splits; newest summer as test matches the live system |
| 2026-11-16 | Atlantic/Gulf split for the coast distances is our own rule, "which sea a coast faces": sea cells inside one hand-drawn polygon (edge roughly along the Keys, then west to 83° W) are Gulf water, all other sea is Atlantic, and a coast cell takes the name of the sea it touches. Indian River Lagoon counts as sea; the Keys are 0 km from both coasts; coasts outside the box are ignored | one polygon, no tracing of the coastline. Natural Earth draws the lagoon as sea, so its mainland shore is the coast (Orlando is 54.5 km from the Atlantic). Each Key touches both waters. Adding coasts outside the box (2° of padding) changes no land cell in the evaluation region |

## Sources

- [NOAA MRMS on AWS](https://registry.opendata.aws/noaa-mrms-pds/)
- [SEVIR storm event dataset](https://arxiv.org/pdf/2207.07033)
- [pysteps nowcasting library](https://gmd.copernicus.org/articles/12/4185/2019/)
- [MeteoSwiss COALITION-3](https://www.meteoswiss.admin.ch/about-us/research-and-cooperation/projects/2020/coalition-3.html)
- [DWD: ML prediction of storm properties (EMS 2023)](https://meetingorganizer.copernicus.org/EMS2023/EMS2023-451.html)
- [Haerter et al., Convective self-aggregation as a cold-pool-driven critical phenomenon](https://arxiv.org/pdf/2002.03474)
- [Haerter et al. 2019, Circling in on convective organization](https://arxiv.org/abs/1810.05518)
- [Khouider, Biello & Majda 2010, A stochastic multicloud model for tropical convection](https://projecteuclid.org/euclid.cms/1266935019)
- [PreDiff (CSI metric on SEVIR)](https://arxiv.org/pdf/2307.10422)
- [DOT AAM National Strategy (Dec 2025)](https://www.transportation.gov/sites/dot.gov/files/2025-12/AAM%20National%20Strategy%202025_508c_251201.pdf)
- [DOT AAM Comprehensive Plan (Dec 2025)](https://www.transportation.gov/sites/dot.gov/files/2025-12/AAM%20Comprehensive%20Plan%202025.pdf)
- [Hackathon rules](https://drreacto.github.io/florida_poly_ai_hackathon/rules.html)
