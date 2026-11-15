# Architecture review: StormLattice

Reviewed Nov 15, 2026, against `docs/design.md` (design, Nov 13 addendum, decision log), `README.md`, the module stubs and the hackathon rules page. No other file was changed.

## Verdict

The idea is sound, and the three decisions you want kept can stay: trained model plus simulation as the core, LLM in a supporting role, SunTrax to Orlando corridor. None of the findings below requires dropping them.

The plan as written does not fit one person in 17 days. Three parts of it would also produce wrong numbers even if finished on time: the lattice applies object-level rates to single pixels, the evaluation cannot attribute a gain to births, and rates trained for 10 minutes are reused out to 60. Each has a small fix.

## Findings ranked by impact

| # | Finding | Smallest fix | Details |
| --- | --- | --- | --- |
| 1 | The rules say "Teams of 2–4 people. No solo entries." Registration closes Nov 30. | Treat as a blocker with your own deadline of Nov 20, not a risk-table row. | §2 |
| 2 | Scope is about three times what 17 days allow. One part of step 1 works; every other module raises `NotImplementedError`; tracking was due today. | Build the minimum claim first (§1), cut the list in §1. | §1 |
| 3 | The two hard rules (LLM, GCP) and four of seven rubric criteria depend on work scheduled last, across Thanksgiving. | Deploy a thin end-to-end page by Nov 22, then improve the forecast behind it. | §1, §5 |
| 4 | Lattice units do not match model units: object deaths and neighbourhood births are applied per pixel. Route risk comes out wrong. | Deaths per object, births per coarse block. | §3.3 |
| 5 | The evaluation cannot support "the gain comes from births", and the CSI comparison favours any probabilistic forecast. | Three extra rows in the ablation table, a probabilistic advection baseline. | §3.4 |
| 6 | Birth labels through Hungarian tracking are the most expensive and noisiest route to the labels. | Label births on the grid as "storm not explained by advection". No tracking. | §3.1 |
| 7 | Frozen rates are trained for the next 10 minutes but drive all 60. | Lead time as a feature, one rate map per lead. | §3.2 |
| 8 | The birth feature list omits the strongest cheap predictor, the weak echo already on radar, and the real sample size is the number of days. | Add sub-threshold reflectivity features; use more days at coarser cadence. | §3.2 |
| 9 | On the demo corridor the flight lasts about 15 minutes, and a 20 km buffer is probably red all afternoon. | The departure window is the product; test the buffer on observed radar. | §2, §5 |
| 10 | The LLM's stated job is partly already done by the data source, and NOTAM API access is by email request. | Narrow the parser to free-text NOTAMs; email the FAA today. | §2, §5 |

---

## 1. Feasibility

### The minimum version that makes the core claim

"Learned storm-birth rates beat advection" needs no tracking, no lattice, no HRRR, no lightning and no SEVIR. It needs six things:

1. **Radar archive.** MRMS composite reflectivity for summer days chosen by a rule (for example every sixth day of June–August), at 10-minute cadence, cropped and regridded to the 2 km grid.
2. **Advection.** pysteps motion and extrapolation, giving the advected storm mask at each lead time.
3. **Grid labels.** A block is a birth at lead L if it holds ≥35 dBZ in the window [t0+L, t0+L+10], held none in the previous 20 minutes, and is not covered by the advected mask. Use 8 km blocks (4×4 cells).
4. **Features at t0.** Distance to each coast, water fraction, land mask, local solar time, day of year, lead time, and the radar state: neighbourhood maximum dBZ, its 10-minute trend, distance to the nearest storm, storm area within 20 and 50 km.
5. **Model.** One XGBoost classifier, calibrated on validation days, giving a birth probability per block per lead.
6. **Score.** Advection plus births against advection alone, and against climatology, at 30, 45 and 60 minutes on held-out days.

That is about five working days and yields the proof chart. The lattice then goes on top as the thing that turns rates into coherent futures for route risk:

- Two states (empty, active), Design A, κ = 0.
- Existing storms are objects from the t0 frame. Each moves by cumulative displacement and dies as a whole with a hazard from an empirical survival table (fraction of advected footprint still ≥35 dBZ at each lead, binned by storm size). No death model yet.
- Births switch on a whole 8 km block with the learned rate for that lead.
- Route risk and the departure window read the members.

### Cut or defer

| Item | Decision | Why |
| --- | --- | --- |
| SEVIR | Cut | 2017–2019 VIL tiles from a different algorithm than live MRMS, few over Florida. The MRMS archive gives the same product you run live. |
| GOES GLM lightning | Cut | Thousands of 20-second files per day. If you want lightning on the map, MRMS carries NLDN density grids in the same bucket. |
| NLCD, 3DEP, coastline curvature | Cut | Coast distances and water fraction come from the Natural Earth shapes cartopy already draws. Elevation is noise in Florida. |
| Hungarian tracking, split and merge logic | Cut | Replaced by grid labels (§3.1). |
| Intensity regression model | Cut | Already optional. |
| Cold-pool state, κ triggers, gust-front layer | Defer to future work | No observation to fit or check them against (§3.3). |
| Exact SSA patch test, mean-field PDE solver | Cut | A closed-form check does the same job in ten lines (§3.3). |
| A* reroute, energy gauge | Cut | The corridor is short; the departure window carries the value. |
| deck.gl 3D columns, ten futures side by side | Cut | A 2D map with image overlays and a time slider is enough. |
| LLM parsing of SIGMETs, METARs, TAFs, PIREPs | Cut | The API returns these decoded (§5). Keep NOTAM free text and the briefing. |
| Spread–skill metric | Cut | No judge will ask; reliability covers calibration. |
| HRRR features | Defer, add as an ablation | Most valuable deferred item. Train the first model without it. |
| XGBoost death model, VIL, echo tops | Defer | Upgrade the survival table if ahead of schedule. |

### Order of work

| Dates | Deliverable | Decision point |
| --- | --- | --- |
| Nov 15–17 | Batch fetch running overnight; regrid; advection baseline; headroom test (§2). FAA email sent. Teammate search. | Headroom number known. |
| Nov 18–21 | Grid labels, feature table, climatology baseline, first XGBoost, first proof chart on validation days. | Does XGBoost beat climatology? Sets the pitch. |
| Nov 22 | Thin page on the VM: observed-radar replay, route card using advection risk, LLM briefing from those facts. | Both hard rules satisfied. |
| Nov 23–27 | Lattice, route risk, departure window, NOTAM parsing. Thanksgiving is Nov 26. | |
| Nov 28–Dec 1 | Test set run once, accuracy panel, video recorded Nov 30, code frozen Dec 1. | |

---

## 2. Weakest assumptions, with early tests

Ranked by how likely each is to sink the submission.

| Rank | Assumption | Why it may fail | Early test | Time |
| --- | --- | --- | --- | --- |
| 1 | A teammate appears by Nov 30. | Solo entries are not allowed. Nothing else matters if this fails. | Post in #find-a-team today; ask the organizers what happens if nobody joins. | 30 min |
| 2 | New storms are a large share of what advection misses. | This is the ceiling of the whole claim, and it is unmeasured. | **Headroom test.** On 3–5 summer afternoons, measure the fraction of observed ≥35 dBZ area at +30, +45 and +60 min that lies more than 10 km from the advected t0 storms. | Half a day once advection runs |
| 3 | The model knows more than "afternoons, inland". | Static geography plus clock time is a location lookup. With 20–30 days the model has seen 20–30 wind regimes. | Fit climatology (birth frequency by block and hour) and a first XGBoost on 10 train and 3 validation days. Compare Brier scores. | First day the feature table exists |
| 4 | Labels mean what you think. | A storm hovering at 34–36 dBZ is born and dies repeatedly. | Plot one day of births on the map; count births per day and the share that last one frame. | 2 hours |
| 5 | The route card is informative on this corridor. | The corridor is about 60 km, so a 150 mph flight takes about 15 minutes. A 20 km buffer covers roughly 3,700 km² of the sea-breeze collision zone. | Compute route risk from **observed** radar on a replay afternoon for buffers of 5, 10 and 20 km. If 20 km is red from 2 pm to 8 pm, pick a buffer that discriminates and say why. | 1 hour, no model needed |
| 6 | Training data arrives in time. | A reflectivity frame is 1.0–1.6 MB and a day at 10-minute cadence is 144 frames, so 60 days is about 13 GB and 8,600 decodes. | Fetch and crop three days, time it, extrapolate. Start the full batch tonight. | 1–2 hours |
| 7 | NOTAM data is available. | The FAA's NOTAM Management Service API is not self-serve; access is requested by email. | Send the request today. Plan to demo on saved NOTAM text if no key arrives. | 10 min |
| 8 | An e2-small runs the live cycle. | It sustains half a vCPU with 2 GB. Each CONUS frame decodes to 24.5 million points before the crop. | Run one full cycle in Docker with `--cpus=0.5 --memory=2g`. | 1 hour once a cycle exists |
| 9 | Live mode behaves in December. | A model trained only on summer storm days has never seen a quiet day or frontal rain. | Run the birth model on a clear December frame and on a cold-front frame. It should predict almost nothing. | 30 min |
| 10 | Demo day infrastructure holds. | Venue Wi-Fi, the VM, the LLM API. | Run the replay with Wi-Fi off and with the LLM key removed (the template briefing should appear). | 30 min |

On rank 8: an e2-medium is roughly twice the monthly price and still fits $50 for three weeks. Check the pricing calculator rather than optimizing for the smaller machine. A 10-minute worker cycle is enough, since the lattice step is 5–10 minutes. Replay mode needs no compute at all if the days are precomputed.

---

## 3. Scientific soundness

### 3.1 Cell tracking and labeling

1. **Tracking is not needed for the birth labels.** Florida afternoon storms are multicell clusters that merge and split constantly at a 35 dBZ threshold, so centroids jump and Hungarian matching produces false births and deaths exactly where the interesting storms are. The grid label in §1 ("≥35 dBZ that advection does not explain") is also the quantity the evaluation scores, so labels and metric agree by construction.
2. **The minimum-area filter is a no-op.** 4 km² is one cell on a 2 km grid. Require several pixels (for example ≥4 in a block) and persistence over two consecutive frames, or one-pixel flicker dominates the positives.
3. **Regridding method matters.** Averaging dBZ from 1 km to 2 km dilutes cores below 35 dBZ. Take the block maximum.
4. **Cadence is unspecified.** MRMS is 2-minute, labels are 10-minute, the lattice is 5-minute. Pick 10 minutes for labels and features; nothing in the claim needs finer.

Connected components at t0 are still useful, as the objects the lattice moves and kills. Only frame-to-frame matching goes.

### 3.2 The XGBoost birth and death framing

1. **Frozen rates go stale.** The model predicts the next 10 minutes from features at t0. Design A then reuses that rate for all twelve steps, so by step six it is applied 50 minutes outside what it was trained for. Train on the target "birth in [t0+L, t0+L+10]" with L as a feature. The lattice gets one calibrated rate map per lead, and Design A becomes legitimate.
2. **The target is a neighbourhood probability, used as a per-cell rate.** "A birth within 4 km" turns one birth into about 13 positive cells. Read as a per-cell propensity, that creates about 13 times too many births. Block-level labels fix it: the label, the rate and the simulated footprint share one unit.
3. **The strongest predictor is missing.** A ≥35 dBZ birth is usually a 15–30 dBZ echo ten minutes earlier. Neighbourhood maximum reflectivity and its trend are free from data you already load, and your `-99 -> -32 dBZ` decision preserved them. Expect these to dominate at 10–20 minutes and geography to matter at 45–60.
4. **Days are the sample size.** Wind regime decides which coast's breeze wins, and each day contributes one regime. Sixty to ninety days at 10-minute cadence beats thirty at 2-minute. Use shallow trees and a large `min_child_weight` so the model cannot memorize where storms happened on particular days.
5. **Choosing only storm days biases the base rate.** Select days by rule and keep the quiet ones.
6. **Calibrate on undownsampled validation data.** Isotonic regression fitted after negative downsampling reproduces the inflated rate.
7. **HRRR, when added, must match what is available live.** Use forecast hours that had been published at t0 (two or more hours after initialization), never analyses. Herbie fetches single variables by byte range.
8. **ROC AUC will flatter.** Ocean at 3 am against land at 3 pm gives a high AUC with no skill. Report Brier skill against the block-and-hour climatology.

### 3.3 The lattice

1. **Deaths are drawn per pixel.** `dies = active & (rng.random(active.shape) < p_death)` erodes a storm one random pixel at a time. For a 100-pixel storm with a 10% per-step hazard, the chance the whole storm is gone after 12 steps is 72% if it dies as an object and about 10⁻¹⁵ if pixels die independently. Since route risk asks whether *any* active pixel is within the buffer, storms effectively never clear the route. Draw one death per object per member.
2. **Births are independent single pixels.** A born cell is 4 km², never grows, and is uncorrelated with its neighbours. That is speckle, not storms, and it is the independence that Step 7 says overstates risk. Block-level births give each new storm a realistic footprint.
3. **With κ = 0 and frozen rates the ensemble adds nothing to the probability map.** Each cell's probability has a closed form from the cumulative hazard. The members earn their place only in route risk and the animation, and only once deaths and births are spatially coherent. That closed form is also the unit test: the ensemble mean must match it. It replaces the SSA patch and the PDE solver.
4. **Iterated advection of a binary field stalls in weak flow.** With nearest-neighbour rounding, any motion under half a cell per step (about 3 m/s at 2 km and 5 minutes) rounds to zero every step. Florida sea-breeze days often have steering flow that weak. Displace each object from t0 by its cumulative displacement.
5. **Cold pools and κ cannot be fitted or checked.** No data layer observes a cold pool, the doc gives no fitting procedure for κ₂ and κ₃, and counting "distinct fronts" by labelling fails when pools merge, which is when collisions happen. The birth model's "recent heavy rain upstream" feature already learns suppression from data. Present the excitable-medium model as the research direction; do not ship a gust-front hazard layer built on an unvalidated state.
6. **Fifty members is noisy at the thresholds.** At 10% risk the sampling error is about ±4 points, so green and amber flicker between runs. Two hundred members cost little once deaths are per object.

The probability-to-rate conversion and the tau-leaping argument are correct.

### 3.4 The evaluation

1. **No row isolates births.** At 60 minutes most t0 storms have died, so pure advection is mostly false alarms and any decay improves it. Add **advection plus deaths only**. Without it, "the gain comes from births" is unsupported.
2. **No row isolates learning.** Add **the lattice driven by climatological birth rates**. The doc's climatology row is a standalone forecast, which does not separate "learned rates" from "having births at all".
3. **The CSI comparison is unfair.** A sharp deterministic forecast that is a few kilometres off is penalized twice, as a miss and a false alarm. A smooth probability map thresholded low wins by over-forecasting. Give advection the same treatment: a neighbourhood-smoothed probability with its threshold tuned on the same validation days. Report frequency bias beside CSI, and lead with Brier skill.
4. **Birth hit rate has no false-alarm counterpart.** Forecasting births everywhere scores 100%. Pair it with the false-alarm ratio.
5. **Demo days must be test days.** If the replay day was in training, the demo is in-sample. Pick replay days from the held-out set, chosen before looking at results.
6. **Report uncertainty by day.** With 10–15 test days, give a bootstrap interval over days and a win count ("better on 11 of 14 days").

---

## 4. Simpler alternatives

Ranked by work saved for result kept.

1. **Grid labels instead of tracking.** Saves about three days and removes the largest source of label noise. Loses storm age as a feature.
2. **MRMS archive only.** One product for training, replay and live, so no domain shift and one fetch path. Saves SEVIR, GLM and most of step 1.
3. **Survival table instead of a death model.** A lookup by storm size and lead, measured from advected overlap. Saves a second model and its calibration.
4. **Direct neighbourhood model as the benchmark.** Train XGBoost on "any ≥35 dBZ within d km at t0+L" and take route risk as the maximum along the route. No lattice. It costs an hour once the feature table exists and tells you honestly what the lattice adds. It is also the fallback if the lattice slips, with the lattice presented as work in progress.
5. **Natural Earth geography.** Half a day instead of three.
6. **Precomputed replay.** Store PNG overlays and risk JSON for two test days. The page then needs a static file server and one LLM call.

---

## 5. Judging

Seven criteria on 70 points; I am assuming 10 each, which the rules page does not state. Four of them (video, pitch, usability, theme fit) score what judges see and hear, and the plan spends most of its time on the other three.

### What will impress non-experts

1. **Prediction, then reality.** Replay at 1 pm with clear skies. Advection shows zero risk for the hour, because there is nothing to move. StormLattice shows the departure window closing. Scrub forward and storms erupt on the corridor. "The classical method scores zero on new storms by construction" is a sentence anyone follows.
2. **The birth-likelihood map** glowing along the collision line before any storm exists, then the observed storms laid over it.
3. **Local stakes.** SunTrax is next door to the judges, and I-4 is their commute. The permanent Disney World flight restriction sits a few miles north of the straight corridor (from memory: FDC 4/3634, 3 nm radius, to 3,000 ft; verify). It is a real NOTAM that was in force on any replay date.
4. **A public URL and a QR code** that work on a judge's phone.
5. **A fail-safe shown live.** Cut the data feed and the card turns to "advisory unavailable, treat as no-go".
6. **Accuracy in plain words.** "When we said 30%, storms came 3 times in 10."

### What they will poke at

| Question | Prepare |
| --- | --- |
| "Is this live?" | Open live mode first: real radar, timestamp, low risk. Then switch to replay under its banner. |
| "Was that day in your training data?" | The answer must be no (§3.4). |
| "How much better is it?" | One number, one chart, with the deaths-only row beside it. |
| "Why do you need an LLM?" | SIGMETs and METARs come decoded from aviationweather.gov, as I recall; check the JSON output. Claiming the LLM parses them invites the question. Free-text NOTAMs and the cited briefing are the defensible role. Show the injection test. |
| "A July storm with December NOTAMs?" | For replay days, pull that day's SIGMETs and METARs from the Iowa Environmental Mesonet archive and label saved NOTAM text as sample data. |
| "Why not the weather app or HRRR?" | HRRR updates hourly and lands about an hour late; existing nowcast products serve en-route altitudes. Cite Recommendation 2.8. |
| "A 20-mile buffer on a 38-mile route?" | The buffer test result from §2, and the operator-set control. |
| "What if it is wrong?" | Advisory only, a person decides, stale data means no-go. |
| Future work (mandatory) | The doc's list is good. Lead with the closed-loop rates and aircraft-reported weather; keep it to 20 seconds. |

Keep Gillespie, contact process, Greenberg–Hastings and propensity out of the pitch. "We roll the dice 200 times" is enough. Do not claim that forecasting storms from colliding boundaries is new; NCAR's Auto-Nowcaster work did that for Florida sea breezes in the 1990s. The learned rates in a stochastic lattice are your contribution.

---

## Fine as-is

- The three core decisions, and the framing on DOT Recommendation 2.8.
- Design A over Design B for the hackathon.
- The 2 km EPSG:5070 grid, and the MRMS flag handling.
- XGBoost as the model, split by whole days, isotonic calibration, test set run once.
- The probability-to-rate conversion and the Poisson assumption behind it.
- Route risk by counting members rather than multiplying point probabilities.
- Stale data means "advisory unavailable, treat as no-go"; timestamps and the LIVE / REPLAY / SIMULATED banner.
- LLM guardrails: schema validation, no numbers of its own, NOTAM text treated as data. Pass the facts as preformatted display strings so the number check does not trip on rounding.
- Replay mode for December, and the offline plus opt-in live test setup.
- The risk table's fallbacks, and the note to search the literature before claiming novelty.
- `storm.alexcozma.dev` answered 200 on `/` on Nov 15, so the domain and TLS exist. `/health` returned 404, so the API is not behind the proxy yet.

## What was checked, and what was not

Checked on Nov 15: the rules page (team size, LLM and GCP requirements, rubric criteria); the public MRMS bucket (archive from 2020-10-14, 650–720 reflectivity frames per day at 1.0–1.6 MB, VIL, echo tops and NLDN density present for 2021, 2025 and 2026); the FAA's NOTAM API access process; the demo URL.

Not checked: decode time per frame, whether the NLDN grids hold usable data over Florida (the files are only a few kilobytes), GCP prices, the aviationweather.gov response format, and the Disney restriction's details. The headroom and model-skill numbers are unknown until the tests in §2 run.
