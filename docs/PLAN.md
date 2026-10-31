# ICARUS plan and task list

Living file: update the status and the decision log in the same PR as the work.
Real analysis table expected around Nov 18; until then C and D build and test on
synthetic data.

## Rules (all tasks)

- Read `src/schema.py` first; every table goes through `schema.validate`.
- Passive objects only (debris, rocket bodies, dead satellites), except the offset ratio.
- Never use BSTAR or air density as a column, outcome or control.
- Day 0 = `storms.main_onset`; event window -10..+20 days.
- Outcome `Y_residual` = MSIS 2.1 residual (m/day); `Y_residual_nrl` for robustness.
- Cluster SEs by `storm_id` with a small-sample correction (20 storms): wild cluster
  bootstrap or CR2.
- Every result comes with its check: placebo, pre-trends or synthetic recovery.
- Every headline result is also reported without the May 2024 and Oct 2024 storms.
- One branch and one PR per task, never push to `main`. Never commit `data/`, `.env`,
  `*.parquet`, `*.csv`. Run `pytest` before committing.

## Done

| PR | What |
|----|------|
| #1 | 5-object sanity check across May 2024 |
| #2 | B1-B2: space weather table and Dst storm catalog |
| #3 | B4-B5 prototype: NRLMSISE-00 / MSIS 2.1 residuals for two rocket bodies |
| #4 | Step 0: shared analysis-table schema |
| #5 | Synthetic generator, causal graph, schema follow-ups |
| #6 | Stacked event study with clean controls (synthetic data) |
| #10 | This planning file |
| #12 | Event-study checks (#7), offset ratio (#8) and high-altitude group / hourly ap (#9) brought to `main`, with the #6-#9 review fixes and the curved UV option |
| #13 | Curved UV error fades above 800 km |
| #14-#16 | Double ML, curved-world result, ap lag sweep |
| #17 | `src/causal_run.py`: every track C result on one table |
| #18 | A2-A4: passive cohort, TLE cleaning, decay intervals (`orbit_intervals_a2_a4.parquet`) |
| #19 | Residual validation: outlier filter, `Y_total`, TLE-blur check, notebook 04 |
| B6 | Physics simulator (`src/simulate.py`) with a known storm boost (PR pending) |
| B4-B5 | Cohort physics and the real analysis table (`src/build_table.py`, notebook 05; PR pending) |

## A: Piyush (orbit data)

- [x] **A2-A4 cohort and cleaning** (#18, follow-up fixes #20): 400-800 km passive cohort, TLE cleaning, intervals
      that never bridge a `main_onset`. First download (#18): 250 objects, 188,560 intervals; checked locally
      Oct 30: no nulls, no duplicate keys, 0 intervals bridge a `main_onset`, all ecc < 0.01.
      The cleaning flagged 0 inclination / RAAN jumps and 0 rolling-median outliers;
      B's Hampel filter on the residual (`physics.remove_outliers`) is the outlier step.
- [x] **High-altitude control group** (#18, 15 objects, 1,040-1,211 km): 10-20 near-circular (ecc < 0.01) passive objects
      at 1,000-1,500 km (old rocket bodies, debris), kept as a separate group. Not
      LAGEOS itself (~5,900 km).
- [x] **A7 precise-orbit check** (required): GRACE-FO-1 May 2024 pilot comparison is
      in `src/precise_orbit.py`; the full Swarm comparison remains an optional extension.

## B: Sonu (space weather and physics)

- [x] **Residual validation** (branch `B-residual-validation`, `notebooks/04_residual_validation.ipynb`;
      two rocket bodies until the cohort exists)
  - [x] Outlier filter on residuals: Hampel, 5-day window, k = 3.5 (`physics.remove_outliers`),
        removes 4.2% of intervals (~1% on pure normal noise).
  - [x] TLE-blurring check: prediction smoothed over 1, 2, 3 days and lagged (best lag -3 h).
        May 2024 over-prediction vanishes with 3-day smoothing; Oct 2024 (41341) survives.
  - [x] Baseline residual across all 20 storms (`physics.storm_baseline_residual`): no
        over-prediction overall (MSIS 2.1 +4.0 m/day, 95% CI -4.8 to 12.8); held-out quiet
        residual ~0 for both objects and models (|t| < 0.6) after the B fix below.
  - [x] Two estimands:
    (a) total storm effect = observed minus MSIS 2.1 run with ap replaced by the
        storm's pre-storm quiet ap (median ap over days -10..-3; ap is global, so one
        value per storm). F10.7 stays at its actual values, so the difference isolates
        the storm from Sun UV;
    (b) model error = observed minus MSIS 2.1 with actual ap (current `Y_residual`).
  - [x] Report both, with and without the May 2024 and Oct 2024 storms.
  - [x] Save (a) as its own column `Y_total` (m/day): optional column in `schema.py`
    (`OPTIONAL_COLUMNS`), built by `physics.quiet_ap_drivers` + `physics.add_total_effect`.
  - [ ] Rerun notebook 04 on the full cohort (with B4-B5 at scale). Notebook 05 covers the
        holdout check, May/Oct residuals and the event-day profile; the smoothing / lag
        checks of notebook 04 are still to rerun on the cohort.
  - [ ] Blur-free outcome for the extreme storms: total altitude lost from a calm day before
        to a calm day after (decision rule below).
- [ ] **B3 object catalog** (`src/objects.py`, branch `B3-object-catalog`): one row per
      object, columns in `schema.OBJECT_COLUMNS`, checked by `schema.validate_objects`.
  - [x] CelesTrak SATCAT (`fetch_satcat`, one JSON request per object, cached in
        `data/raw/satcat_records.json`): `object_type` (debris / rocket body / payload /
        unknown), `launch_date`, `decay_date`.
  - [x] DISCOSweb (`fetch_discos`, token `DISCOS_TOKEN` in `.env`, API v2, 100 objects
        per request, 3 s between requests, 429 waited out via Retry-After, cached in
        `data/raw/discos_objects.json`): `mass_kg`, `xsect_avg_m2` (xSectAvg),
        `area_to_mass` = A/m, `drag_sensitivity_catalog = 2.2 * area_to_mass`.
  - [x] `detect_maneuvers(intervals)`: lasting semi-major-axis rises above the decay trend
        and the object's own noise; `is_passive` / `is_active` (debris and rocket bodies
        passive; payloads passive only if searched and no maneuver found; payloads without
        intervals count as active).
  - [x] `build_objects(norad_ids)` -> `data/processed/objects.parquet` (maneuver search on
        `orbit_intervals_a2_a4.parquet` by default).
  - [x] `compare_drag(measured, catalog)`: Spearman rho and log-log plot, fitted B vs
        catalog, rocket bodies with both.
  - [x] Tests (`tests/test_objects.py`, inline records, fake HTTP session); the two
        download tests skip offline / without `DISCOS_TOKEN`.
  - [ ] Run locally on the cohort (CelesTrak and DISCOSweb were unreachable from the
        build container): `build_objects` on the 265 cohort IDs, check DISCOSweb field
        names against a real reply, then `compare_drag` on the B4-B5 fits.
  - [ ] Notebook 06 (notebooks 05 are taken).
- [x] **B6 physics simulator** (`src/simulate.py`, branch `B6-physics-simulator`):
      synthetic orbits decaying through pymsis MSIS 2.1 density with a storm boost of
      known size, TLE-style noise that grows during storms, output validated with
      `schema.validate`. Done:
  - [x] Space weather: real hourly frame + storm catalog if passed, otherwise synthetic
        hourly ap / f107 / f107_81d with storms planted at given dates (default one
        every 25 days).
  - [x] Objects: passive, near-circular, 400-800 km, random true B; `n_high_alt`
        objects at 1,000-1,500 km (default 3) with catalog B, `drag_source` =
        "catalog" and `Y_log_ratio` NaN.
  - [x] Truth: da/dt = -B rho (1 + boost(event_day)) sqrt(mu a), rho from MSIS 2.1 on
        a circular-orbit track every 20 min, inputs passed explicitly (no download).
        Default boost 0.5 on days 0..2, gone by day 5; negative boost allowed.
  - [x] Fake TLEs: irregular epochs (~1/day, ~2/day in storms, one at each
        main_onset), a noise 20 m quiet and 3x in storms, optional trailing 1-3 day
        smoothing (`smooth_days`).
  - [x] Pipeline: `orbit.decay_rates` -> driver (same samples, density and formula as
        `physics.density_drivers`) -> `physics.label_intervals` -> quiet fit of B ->
        `physics.add_predictions`; extra columns `true_effect`, `true_decay`,
        `obs_decay`, `true_drag_sensitivity`, `drag_source`, `role`.
  - [x] Tests (`tests/test_simulate.py`, ~25 s): no noise -> prediction follows the
        truth (median error < 1%); boost 0 -> residual ~0; held-out quiet residual ~0;
        planted boost (days 0..2) inside the event-study 95% interval; noise larger
        in storms; negative boost; smoothing delays the storm; schema columns.
- [x] **B4-B5 at scale** (branch `B4-B5-cohort-physics`, `src/build_table.py`,
      `notebooks/05_cohort_physics.ipynb`): `python -m src.build_table` rebuilds
      `data/processed/analysis_table.parquet` (199,580 rows, 265 objects, validated).
  - [x] Drivers for NRLMSISE-00, MSIS 2.1 and MSIS 2.1 with quiet ap, cached in chunks of
        10 objects (`data/processed/physics_chunks/`, resumes after a crash). Benchmark:
        3,870 intervals in ~50 s; the cohort took ~37 min.
  - [x] Climb intervals kept (2.2% of main intervals, `climb_flag` column); Hampel filter on
        the MSIS 2.1 residual sets `tle_ok = 0` (5.5%, incl. 64.5% of the climbs).
        Check without climbs: B changes by 0.4% (median), short-run effect -14.6 vs -14.1 m/day.
  - [x] Drag sensitivity: ratio of sums on clean quiet fit intervals, 500-draw block
        bootstrap by week (`drag_se`). Positive and stable for all 250 passive objects
        (SE / B median 3%, max 8%). Unstable rule below.
  - [x] High-altitude group: `Y_log_ratio` NaN. Their quiet fit is tight on real data
        (SE / B ~4%, held-out residual 0.03 m/day, t = 0.4), so they keep it
        (`drag_source = "fit"`); an unstable one falls back to the main-cohort median B
        (`"cohort_median"`) until the B3 catalog exists.
  - [x] Gate passed: held-out quiet residual -0.13 m/day (weekly-block SE 1.07) pooled over
        50,674 intervals vs mean decay -50.6 m/day; per-object t SD 0.82, 6 of 250 with |t| > 2.
  - [x] Smoke test of `src.causal_run` (PRELIMINARY, see decision log). It now skips the
        offset ratio when the table has no active satellites (needs B3).

## C: Siddhant Anil (event study and offset ratio, `causal-event-offset`)

- [x] **Synthetic additions** (#9, on `main` via #12): `n_high_alt` (1,000-1,500 km
      group) and `return_hourly` (hourly frame in the format of
      `omni_io.load_spaceweather()`, not in `table.attrs`).
- [x] **Event study** (#6, on `main`): stacked, clean controls, CR1 + wild cluster
      bootstrap by storm.
- [x] **Event-study checks** (#7, on `main` via #12): planted recovery, placebo-based
      pre-trend test, 500 placebo dates, with/without May/Oct 2024.
- [x] **Offset ratio** (#8, on `main` via #12): matched twins, station keeping removed
      via days -10..-3, ducking and recovery windows, debris placebo.
- [x] **Follow-up PR to `main`** (#12) bringing #7-#9 in, with these fixes:
  - `event_study(exclude_storms=...)` must still treat the excluded storms as
    contaminating (pass them to `build_stacks(..., others=...)`). Now 2,713 rows
    from inside the excluded storms become clean controls (synthetic, seed 0).
  - `main_sample()` must drop rows above 1,000 km (6,140 of 37,312 rows got in with
    `n_high_alt=15`); add `high_alt_sample()` for the control.
  - `Y_log_ratio` NaN for the high-altitude group (it comes out at a median of ~11).
  - `offset_ratio(..., outcome=)` argument. On real data the storm loss must use the
    total storm effect (`Y_total` from B), not `Y_residual`: `Y_residual` is model
    error, and MSIS already follows most of the storm.
  - Docstring nit: `f107_81d` is no longer a yearly wave.
- [x] **Sensitivity runs** (#12): event study with `contam=(-3, 10)` as well as the default
      `"active"` (our headline is slow recovery, which can leak into the next storm's
      pre-period); offset ratio with a 5-day recovery window as well as 10 days.
- [x] **Curved confounder option** in `make_synthetic` (#12, #13): UV model error that
      grows with altitude up to 800 km, fades above it, and saturates at high F10.7.
      Default stays linear so existing tests don't move.
- [x] **Real-run script** (`src/causal_run.py`): every C result on one table, for both
      estimands (`Y_total`, `Y_residual`), with/without May/Oct 2024, the `(-3, 10)` and
      5-day sensitivity runs, the high-altitude negative control and the offset ratio
      on `Y_total`. Writes CSVs and `summary.md` to `results/causal`. Tested on
      synthetic data; on Nov 18 run `python -m src.causal_run`.
- [ ] **Paper**: intro, data/physics setup, DAG, event study, offset ratio. Draft
      written on synthetic results; real numbers go in from `results/causal/summary.md`.

## D: Sidhant Holla (double ML and vulnerability map, `causal-dml-forest`)

- [ ] **DML** (`src/dml.py`): outcome `Y_residual`, confounders `schema.CONFOUNDERS`,
      XGBoost/CatBoost with cross-fitting. Naive regression vs DML side by side,
      first on synthetic data (naive should be fooled by F10.7), then on real data.
  - Treatment = interval-mean ap = `T_ap_int / dt_days` (nT). Raw `T_ap_int` mixes
    interval length into the effect (storm intervals ~0.5 day, quiet ~1 day).
  - Weight by `dt_days` (residual noise scales with 1/sqrt(dt)).
  - Cross-fitting folds by storm / time block, never by row: each storm's whole
    -10..+20 window in one fold; quiet periods in their own time blocks.
  - Wild cluster bootstrap on the final residual-on-residual stage, nuisance fits fixed.
  - Helper (nuisance) models, rule fixed before any real-data run: for the ap model
    and the residual model separately, try a straight line, a cautious XGBoost
    (shallow, heavily regularised) and a flexible XGBoost; keep the one with the best
    prediction on held-out storms, measured on the same storm-blocked folds as the
    cross-fitting (never random rows). Report the estimate under all three ap models.
    Why: a flexible ap model had negative held-out R² (storm timing is close to
    random once Sun and season are known), its errors added noise to the leftover ap
    and pulled the estimate ~8% toward zero (coverage 41/50).
  - Pass bar: the whole pipeline, selection included, covers the truth in at least
    45 of 50 synthetic worlds, in both the linear and the curved-confounder world.
    If naive and DML agree on real data, report it as robustness.
  - ap lags 0 / 6 / 12 / 24 h: main lag fixed in advance, all four reported. Main lag = 0 h
    (`dml.MAIN_LAG_HOURS`, `dml.lag_sweep`).
  - Status: passes in both synthetic worlds (selection in the loop, true-leftover-ap
    target): linear 48/50, curved 49/50. Lag sweep built. Still to do: real data.
- [ ] **Refutations**: high-altitude group (>1,000 km) shows no effect, tested as
      equivalence (CI inside a pre-set band around zero), not just "not significant";
      it also catches storm-time TLE noise posing as decay. Random common cause leaves
      the estimate unchanged; shuffled storm timing gives zero; with and without
      May/Oct 2024.
- [ ] **Vulnerability map** (`src/vulnerability_map.py`): CausalForestDML by altitude x
      inclination x `drag_sensitivity`, heatmap and feature importance. Must recover
      the planted differences on synthetic data and be stable across 5 seeds.
- [ ] **Paper**: DML method, refutations, vulnerability map, discussion/limitations,
      related work (Parker & Linares; CosmicDance / CosmicDancePro; Oliveira, Zesta &
      Garcia-Sage). What is new here: population-wide, causal, separates the storm
      from Sun UV, model-error and total-effect estimates, validated on synthetic
      ground truth.

## Decision rules

- Post-storm over-prediction survives the smoothing check: headline finding, "models
  recover too slowly after storms".
- It disappears after smoothing: TLE blurring caused it. Switch the outcome to total
  altitude lost from a calm day before the storm to a calm day after.
- A result only holds with May/Oct 2024 included: report it as driven by extreme
  storms, don't generalise.
- Storm windows overlap: drop the contaminated windows, or use only clean controls in
  the stacked event study.
- Held-out quiet residual isn't ~0: stop and fix drag sensitivity first.
- Drag sensitivity is unstable (fixed before the cohort run, `build_table.drag_bootstrap`)
  if the object has fewer than 30 quiet fit intervals, its weekly block-bootstrap SE is
  more than 25% of B, or its bootstrap 95% interval reaches zero. Unstable objects are
  flagged (`drag_unstable = 1`), not dropped; a high-altitude object that is unstable
  gets the main-cohort median B.

## Decision log

- **Oct 30, B4-B5 cohort physics (Sonu)**: climb intervals are kept (dropping only upward
  noise would make decay look too fast); Piyush's `tle_ok` failed them, so the table's
  `tle_ok` is the TLE endpoint check plus the Hampel filter, and `climb_flag` is kept for
  the with/without check. High-altitude objects keep their quiet fit (tight on real data,
  unlike the simulator) rather than a placeholder B: a cohort-median B is ~6x too large for
  most of them and would fake a storm effect in the negative control. Smoke-test numbers
  (PRELIMINARY, 12 clean storms): total storm effect `Y_total` days 0-2 -14.1 m/day
  (95% CI -19.8 to -8.4, placebo p < 0.002, pre-trend p = 0.41), -13.8 without May/Oct
  (`key_storm_ids` also drops the moderate May 2 storm), -10.1 with the (-3, 10) rule;
  model error `Y_residual` +0.2 (CI -3.2 to 3.2, placebo p = 0.88); high-altitude control
  `Y_total` -0.7 (CI -1.4 to 0.1). May/Oct 2024: MSIS 2.1 over-predicts days 0..+3 for
  almost every object (median +12 / +15 m/day), consistent with notebook 04.
- **Oct 30, B3 object catalog (Sonu)**: payloads (and unknown types) count as passive only
  if their decay intervals were searched and no maneuver was found; with no intervals they
  count as active, so nothing unchecked enters the passive sample. A maneuver = median a
  over the next 2 days minus median over the previous 2 days, after adding back the
  expected decay (median of ~1-day rates within +-7 days), above max(0.2 km, 5 x noise);
  noise from the upper half of that statistic, since storms only push it down. Synthetic
  check (100 seeds): no false alarms with or without 2-day storms, single outlier TLEs
  ignored, 0.3 km rises found 100/100 at 50 m/day decay, 1 km rises 99/100 at 400 m/day
  decay with 40 m scatter. Catalog drag uses Cd = 2.2 and DISCOSweb's average
  cross-section; compared with the fitted B by rank (Spearman) only.
- **Oct 30, A2-A4 orbit pipeline (Piyush)**: selected 250 passive 400-800 km objects
  with coverage on at least 8 of 9 Space-Track snapshots, plus 15 separate 1,000-1,500
  km high-altitude controls. The follow-up selection is spread across four 100 km bands
  (39/63/82/66 main objects) and three objects in each high-altitude band. The revised
  Apr 2024-Apr 2025 download yielded 203,954 deduplicated TLEs and 198,388 intervals;
  seven gaps over five days and 5,294 onset-bridging intervals were omitted. The
  three-times-robust-TLE-noise climb rule flagged 416 intervals; the tighter point-level
  diagnostic flags 15 TLEs (0.0074%). `notebooks/05_orbit_cleaning_checks.ipynb` records
  the before/after plot and May 2024 check; generated data remains under ignored `data/`.
  First download (#18, before the follow-up): 205,145 deduplicated TLEs and 199,580 adjacent
  intervals; 5,300 onset-bridging intervals were omitted and 4,705 passive-climb intervals
  were flagged (`tle_ok = 0`). Cleaning thresholds remain explicit parameters for review
  and sensitivity checks; raw TLEs and reports are under ignored `data/` for Drive sharing.
- **Oct 30, B6 simulator (Sonu)**: with TLE noise, a near-zero interval next to an
  inserted main_onset epoch made the day -1 reference row noisy enough to shift every
  event-study coefficient, so epochs closer than 0.2 day to a main_onset are dropped.
  With 6 storms the wild-bootstrap interval missed zero on null data; the recovery
  test uses 300 days (11 storms) and the days 0..2 short-run effect. The quiet fit of
  B is poor above ~700 km (decay of a few m/day vs ~30 m/day TLE noise per interval),
  as on real data; it adds little to the residual in m/day there.

- **Oct 31, A7 precise-orbit check (Piyush)**: compared the TLE mean semi-major axis
  against 19 daily GRACE-FO-1 reduced-dynamic precise-orbit points for May 1-20, 2024.
  After removing the mean-element/osculating level offset, correlation is 0.907,
  centred RMSE is 0.097 km, and the linear decay-slope difference is 0.0025 km/day.
  The flagged-TLE event report has 9 storm-interval flags, 5 post-event-window flags,
  and 1 calm-time flag; this is retained as a diagnostic rather than silently dropping
  storm-time observations.

- **Oct 27, DML review points (Holla)**: high-altitude negative control added (A, B, C, D
  tasks above); hourly ap returned as a space weather frame, not via `attrs`; DML
  treatment is interval-mean ap in nT, described the same way in the paper; DML folds
  and clusters by storm / time block.
- **Two estimands (Siddhant)**: total storm effect vs model error, both reported, with
  and without May/Oct 2024.
- **Oct 27, review of #6-#9**: clean-control default `contam="active"` accepted (the
  strict -10..+20 rule leaves 0 of 20 storms), with a `(-3, 10)` sensitivity run;
  placebo p-values are the main inference for the event study (rows can sit in two
  stacks); offset ratio headlines the ducking share, recovery share reported as
  imprecise; offset-ratio storm loss to use the total storm effect. Fixes listed
  under C.
- **Oct 27, DML helper models (Holla)**: choose each helper model by held-out-storm
  prediction from {straight line, cautious XGBoost, flexible XGBoost}; departs from
  "XGBoost/CatBoost only" because the data picks the model. Coverage test reruns
  with the rule in the loop. Curved confounder added to the synthetic world.
- **Oct 27, DML 50-seed result (Holla)**: on `main`'s synthetic data (linear confounding),
  the 95% CI covers the true-leftover-ap target in 48/50 seeds (bias -0.3%); naive 9/50
  (+149%), linear controls 45/50 (+0.5%). Fixed ap model: straight line 48/50, cautious
  XGBoost 46/50, flexible XGBoost 45/50 (-4.2%). Selection picked XGBoost for the
  residual in 50/50 seeds and a straight line for ap in 45/50. Rule is final for the
  linear world; the curved world is still to run.
- **Oct 27, DML curved world (Holla)**: with `uv_error_kind="curved"`, DML covers the
  target in 49/50 seeds (bias -0.7%, sd 4.7%); linear controls 44/50 (bias -0.7% but
  sd 10.7%, below the bar); naive 7/50 (+111%). Fixed ap model: straight line 49/50,
  cautious XGBoost 42/50, flexible XGBoost 44/50 (-7.2%). The selection rule is final.
  `sun_share` renamed `adjustment_share` ("share of the naive slope removed by
  adjustment", all confounders).
- **Oct 27, DML main ap lag (Holla)**: fixed at 0 h before any real-data run: it is
  the window the decay is measured over, and MSIS already uses the ap history, so the
  residual is the model's error over that window. 6, 12 and 24 h are reported
  alongside, all on the same rows (those with ap at the longest lag; the first day of
  the series drops out). Synthetic check (no lag planted, seed 0): the estimate is
  largest at 0 h and fades with lag (linear world -0.0187, -0.0179, -0.0132, -0.0044
  m/day per nT; planted -0.0188). The 6 h estimate stays close because ap 6 h apart is
  strongly correlated (r = 0.74). On real data, if a later lag beats 0 h, report it as
  MSIS mistiming the storm response (it links to the slow-recovery finding), not as a
  reason to switch the main lag (Sonu, #16 review).
- **Oct 30, residual validation (Sonu)**: drag sensitivity B is now the dt-weighted ratio of
  sums, sum(observed * dt) / sum(driver * dt) on the quiet fit intervals, not the median of
  observed / driver. After outlier removal the median left NORAD 41341 with a held-out MSIS 2.1
  residual of -4.6 m/day (t = -2.2, ~3% low B); the ratio of sums gives |t| < 0.6 for both
  objects and models. Verdict on the post-storm over-prediction: **extreme-storm driven**
  (absent across the 20 storms, +1.2 m/day without May/Oct), and TLE blur in May 2024 (gone with
  3-day smoothing); Oct 2024 partly survives. So the headline is not "models recover too slowly";
  for the extreme storms add total altitude lost calm-day-before to calm-day-after.
- **Oct 31, A7 precise-orbit follow-up (Piyush)**: added orbit-period averaging, daily
  decay-rate comparison, and a ±3-day lag sweep. GRACE-FO-1 October 2024 has its
  strongest agreement at zero lag (r = 0.70). For Swarm-A, the period is now estimated
  from successive radius minima (93.83 min), duplicate timestamps across daily files are
  removed, and states are re-averaged over actual revolutions; the alternating-sign
  artifact is gone. Swarm remains a supporting cross-check pending review of the product.
- **Oct 31, active-satellite intervals (Piyush)**: selected a coverage-balanced payload
  cohort (25 Starlink + 25 other payloads), detected station-keeping activity, and built
  cleaned decay intervals for the offset-ratio analysis. The local run found 14 active
  objects and 10,840 active intervals; no intervals crossed an onset or a >5-day gap.
  The reproducible entry point is `src/active_cohort.py`; raw TLEs and derived tables stay
  ignored under `data/`.
