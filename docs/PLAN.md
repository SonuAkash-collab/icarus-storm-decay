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
| review | First-review figure pack: `python -m src.review_figures` -> `results/review/` (7 figures + SUMMARY.md; PR pending) |

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
  - [x] Rerun notebook 04 on the full cohort (branch `B-extreme-storm-check`,
        `src/extreme_check.py`, `notebooks/09_extreme_storm_check.ipynb`): 250 passive objects,
        20 storms, both models, smoothing 1-3 days, lag. No general post-storm over-prediction
        (MSIS 2.1 +0.8 m/day, CI -5.8 to 7.4); best lag -3 h (negligible). May 2024 does not
        vanish with 3-day smoothing on the cohort (+14.8 -> +6.9 m/day), Oct 2024 +34.6 -> +28.4.
  - [x] Blur-free outcome for the extreme storms (`physics.calm_to_calm`): altitude lost from
        the last calm day in days -10..-3 to the first calm day after recovery_end, observed vs
        MSIS 2.1, onset gaps refilled (`physics.fill_onset_gaps`). Oct 2024 obs/pred 0.86 (real
        overshoot, all altitude bands); May 2024 0.91 raw but ~0 m against the model's local
        quiet error (timing, not total); Dec 2024 1.19 (under-prediction). Review fig 8.
- [x] **B3 object catalog** (`src/objects.py`, branch `B3-object-catalog`): one row per
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
  - [x] First local run (branch `B3-local-run`, `notebooks/06_objects.ipynb`): field names
        checked against raw DISCOSweb and SATCAT replies (no change needed), 265/265 objects
        found in both. 118 have catalog A/m (116 rocket bodies of 116, 2 debris of 149);
        the cohort has no payloads, so 0 flagged active. Rocket bodies: Spearman rho = 0.81
        (n = 116, 0.82 for the 109 in 400-800 km). Next: join `objects.parquet` to the table.
  - [x] Tests (`tests/test_objects.py`, inline records, fake HTTP session); the two
        download tests skip offline / without `DISCOS_TOKEN`.
  - [x] Run locally on the cohort (#25): `build_objects` on the 265 cohort IDs, DISCOSweb
        field names checked, `compare_drag` on the B4-B5 fits.
  - [x] Notebook 06 (#25).
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
- [x] **Active satellites in the table** (branch `B-actives-join`, `notebooks/07_actives.ipynb`):
      `python -m src.active_cohort` then `python -m src.build_table` (appends them when the
      active files exist; `--no-actives` for the passive table only).
  - [x] `src/active_cohort.py` CLI: payload snapshots on the 9 sample dates (cached in
        `data/raw/active_candidates/`), 25 Starlink + 25 other payloads selected separately
        with `select_active_cohort`, TLE history, cleaning, intervals. 50 objects, 39,218
        intervals; 26 active (25 Starlink + SURCAL 150B), 19,900 active intervals.
  - [x] `build_objects` on cohort + actives -> `objects.parquet` (291 objects; maneuver search
        now reads the active intervals too). Starlink A/m from DISCOSweb (v1.0 0.052, v1.5
        0.036, v2 mini 0.046 m^2/kg).
  - [x] Physics: same drivers (`data/processed/physics_chunks_actives/`), B = 2.2 x A/m x
        fitted / catalog ratio of rocket bodies within +-50 km (`catalog_scale`, 0.71-0.77
        at Starlink altitudes; `drag_source = "catalog_scaled"`). No Hampel filter for actives
        (it would delete the burns). `Y_total`, `Y_residual`, `Y_residual_nrl` as for passives,
        `Y_log_ratio` NaN. Per-object B and flags in `data/processed/active_drag.parquet`.
  - [x] `free_drag` flag (`build_table.free_drag_flags`): rows in a 21-day window losing
        > 1 km and >= 10% of the free-drag loss; left out of the offset ratio
        (`offset_ratio._clean_actives`). 17 of 26 objects have flagged rows (7,065 of 19,900
        rows); 5 fully excluded (SURCAL 150B + 4 Starlinks deorbiting all year).
  - [x] `causal_run` reports instead of failing when no active has a twin
        (`offset_ratio.match_diagnostics`, `offset_match_diagnostics.csv`).
  - [x] Twin offset ratio NOT estimable with the current passive pool (decision log, Oct 31);
        replaced by the model-based offset ratio below.

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
- [x] **Real run, event study** (Oct 31, `analysis_table.parquet` of Oct 30, 500 placebo
      draws, 999 bootstrap draws; reproduces Sonu's smoke test). Main sample 178,107 of
      199,580 rows (high-altitude 5.5%, failed TLE cleaning 5.2%); 12 of 20 storms clean.

      | Days 0-2 (m/day) | `Y_total` | `Y_residual` |
      |---|---|---|
      | All storms | -14.06 (CI -19.75 to -8.40) | +0.15 (CI -3.16 to 3.19) |
      | Placebo dates p (main inference) | < 0.002 (0 of 500; placebo mean 0.36, SD 1.27) | 0.88 |
      | Pre-trend p (placebo) | 0.41 | 0.30 |
      | Without May/Oct 2024 (storms 3, 4, 11; 9 storms) | -13.85 (CI -16.63 to -10.87) | -2.22 (CI -4.35 to 0.01) |
      | `(-3, 10)` rule (9 storms) | -10.09 (CI -14.02 to -6.22) | +1.42 (CI -2.52 to 5.38) |
      | High-altitude control (15 objects) | -0.69 (CI -1.42 to 0.07) | -0.17 (CI -0.68 to 0.34) |

      Figures: `results/event_study_Y_total.png`, `results/event_study_Y_residual.png`.
      Day 0 carries most of the total effect (about -30 m/day), day 1 about -12, back near
      0 by day 2; model error is -9 on day 0 and slightly positive on days 1-2.
- [ ] **Real run, offset ratio**: waits for the active satellites to join the table
      (25 Starlink + 1 other); question 3 is then effectively about Starlink.
- [x] **Model-based offset ratio** (branch `C-offset-model`, `src/offset_model.py`,
      `notebooks/08_offset_model.ipynb`, in `python -m src.causal_run`): each active satellite vs
      its own physics-predicted storm loss (MSIS 2.1 actual ap minus quiet ap, catalog-scaled B),
      observed loss vs its own days -10..-3 baseline, ducking + 10-day recovery windows, all 20
      storms, offset = 1 - observed / predicted, storm-clustered wild bootstrap (Webb). Passive
      model-error correction by storm class x 100 km band; free-drag Starlink placebo with
      own-fitted B; B-scale sensitivity. Rows in a 21-day window with > 1 km net altitude change
      (orbit raises, shell changes) are left out. Result in the decision log (Oct 31).
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
    target): linear 48/50, curved 49/50. Real run done (`python -m src.dml_run`, results
    in the decision log, Oct 31); its checks are the refutations below.
- [x] **Refutations** (`src/dml_refute.py`, real run Oct 31, decision log): high-altitude group (>1,000 km) shows no effect, tested as
      equivalence (CI inside a pre-set band around zero), not just "not significant";
      it also catches storm-time TLE noise posing as decay. Random common cause leaves
      the estimate unchanged; shuffled storm timing gives zero; with and without
      May/Oct 2024.
- [x] **Model error by storm class** (`src/storm_class.py`, #40): per-storm change with an exact
      permutation test (Oct 31) and the same in DML, per-class final slopes (Nov 7); decision log.
- [x] **Vulnerability map** (`src/vulnerability_map.py`, real run Oct 31, decision log): CausalForestDML by altitude x
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
- **Oct 31, B3 first run (Sonu)**: fitted vs catalog B for rocket bodies has rho = 0.81, but
  the 116 bodies share only 19 distinct catalog A/m values (15 stage types), so it is
  closer to a check on about 15 stage types than 116 independent points; report it so.
  Fitted B is about 0.8 x the catalog (median), and fitted / catalog grows with altitude
  (Spearman 0.56), consistent with an orientation-averaged cross-section and Cd = 2.2 as a
  convention; only the ranking is used as validation. Debris has no usable catalog A/m
  (2 of 149), so debris B rests on the fit alone.
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
  artifact is gone; **Swarm-A zero-lag correlation is r = 0.457** after actual-period
  averaging. Swarm remains a
  supporting cross-check pending review of the product.
- **Oct 31, active payload guard (Piyush)**: Starlink names still provide the operational
  fallback, but objects with >=21 days of uninterrupted negative decay and no positive
  maneuver are flagged `free_drag_flag` and excluded from operator-response counts.
- **Oct 31, active-satellite intervals (Piyush)**: selected a coverage-balanced payload
  cohort (25 Starlink + 25 other payloads), detected station-keeping activity, and built
  cleaned decay intervals for the offset-ratio analysis. The local run found 14 active
  objects and 10,840 active intervals; no intervals crossed an onset or a >5-day gap.
  The reproducible entry point is `src/active_cohort.py`; raw TLEs and derived tables stay
  ignored under `data/`.
- **Oct 31, actives in the table (Sonu)**: the active cohort was rebuilt with
  `python -m src.active_cohort` (Piyush's files were not shared): 25 Starlink + 25 other
  payloads, 26 active = 25 Starlink + SURCAL 150B, as in Piyush's run. SURCAL 150B (1967
  calibration satellite) is a false-positive active: its 2 "maneuvers" sit on a steady
  ~300 m/day decay, and the free-drag flag removes all its rows. So question 3 is a Starlink
  study. Active drag sensitivity = catalog x the median fitted / catalog ratio of rocket
  bodies within +-50 km (0.71-0.77), not a straight-line fit in altitude (one stage type at
  600-650 km pulls that to ~0.5). Satellites that stop station keeping are flagged per row,
  not per object (shell changes leave good periods), with a 10% lower bound on observed /
  free-drag loss because deorbiting Starlinks decay at only 0.2-0.5 of the catalog rate.
  **The offset ratio is not estimable with the current passive pool**: no passive object
  lies near the Starlinks (0 at 440-570 km and 38-58 deg; nearest twin 1.03-1.34 SD vs the
  0.5 caliper), and the neighbouring-storm rule leaves 3 of 20 storms (Jun 27, Nov 8, Dec 31
  2024; May and Oct 2024 drop out). Forcing pairs with a 1.5 SD caliper gives SMD -1.8
  (altitude) and -1.4 (inclination) and an offset ratio of -4.7 (95% CI -65 to 120): not a
  result. Debris placebo (20 sets): ducking share mean -0.04 (SD 0.16), offset ratio mean
  0.10 (SD 0.45). Passive results unchanged by the actives (`Y_total` days 0-2 -14.06,
  `Y_residual` +0.15). Next: passive twins at 450-560 km and 40-56 deg (task A), or a
  twin-free counterfactual (the active's own catalog-B physics prediction); both are
  design changes for the team to decide.
- **Oct 31, model-based offset ratio (Sonu)**: the twin design has no twins, so each active
  satellite is its own control and MSIS 2.1 (actual minus quiet ap, catalog-scaled B) gives the
  predicted storm loss. Decided before the numbers: all 20 storms, ratio of sums, Webb wild
  bootstrap by storm, with / without May and Oct 2024, passive model-error correction by storm
  class and altitude band. Added after a first run: rows in a 21-day window with > 1 km net altitude
  change are dropped (orbit raises of STARLINK-30105 etc. swamped the recovery window by +-20-37 km),
  and placebo satellites must decay at <= 1.5x the catalog free-drag rate (1010 / 1013 / 1015 /
  30264 / 30277 are thrust down at 3-22x). **Result (21 Starlinks, 20 storms, 282 pairs): total
  offset 1.19 (95% CI 0.97 to 1.42), corrected 1.18 (0.98 to 1.38), without May / Oct 2024 1.39
  (0.89 to 1.92)**; ducking window 1.21 (1.09 to 1.34), ducking part 1.07, recovery part 0.12
  (-0.10 to 0.33). B scale 0.75x / 1.25x: 1.25 / 1.15. Reading: Starlinks lose no more altitude
  around storms than in their baseline, i.e. they undo the whole predicted loss, during the storm.
  Placebo (11 free-drag Starlinks, own B): ducking 0.33 (CI -0.67 to 1.29), total 1.47 (CI -7.7 to
  10.3, drifting baselines of deorbiting satellites): compatible with 0 but weak; with catalog B
  ducking 0.65 (0.13 to 1.15), which shows that a wrong B scale fakes an offset. Caveats: one
  operator, catalog B, overlapping windows (storm 3's recovery window holds May 10), weak placebo.
- **Oct 31: one free-drag rule (build_table.free_drag_flags); active_cohort guard is informational only.**
- **Oct 31, extreme-storm check (Sonu)**: verdict **mixed**. Rules: an over-prediction is a real
  model overshoot only if the days 0..+3 change survives 3-day smoothing (95% CI over objects
  above 0) AND the calm-to-calm loss is smaller than predicted against both baselines (the
  object's pre-storm error, days -10..-3 after the previous storm's recovery, and its mean error
  on quiet intervals within +-30 days). Oct 2024 passes both (+28.4 m/day smoothed; +137 / +169 m,
  obs/pred 0.86). May 2024 survives smoothing (+6.9) but not calm-to-calm (-75 / -1 m): MSIS was
  already over-predicting before the storm (+11.6 m/day) and the running error gives part of the
  day 0..4 excess back after day 6, so it is timing. Dec 2024 is under-predicted (obs/pred 1.19).
  Calm day = every 3-h ap < 27; 13 of 20 storms have a calm day before and after (the March-April
  2025 run does not). Pushing the calm day after 2-3 days later changes nothing (the simulator
  shows this removes a 2-day blur). Note: Oct 2024 is "severe" in the catalog, not "extreme".
  Headline stays: no general model error; extreme-storm error is storm-specific.
- **Oct 31, DML real run (Holla)**: `python -m src.dml_run` on `analysis_table.parquet`
  (Oct 30; main sample 178,107 rows, 38 clusters = 20 storms + 18 quiet weeks), 999
  bootstrap draws, main lag 0 h. Effects in m/day per nT of interval-mean ap:

  | | `Y_total` | `Y_residual` |
  |---|---|---|
  | Naive | -0.395 (CI -0.543 to -0.255) | -0.060 (CI -0.158 to 0.036) |
  | Linear controls | -0.392 (CI -0.535 to -0.261) | -0.054 (CI -0.150 to 0.040) |
  | **DML** | **-0.393 (CI -0.526 to -0.266)** | **-0.053 (CI -0.142 to 0.043)** |
  | DML, ap model cautious / flexible XGBoost | -0.373 / -0.304 | -0.045 / -0.048 |
  | Share of naive slope removed by adjustment | 0.01 | 0.13 |
  | Lag 0 / 6 / 12 / 24 h | -0.39 / -0.33 / -0.21 / -0.06 (24 h CI covers 0) | -0.05 / 0.00 / +0.08 / +0.11 (24 h CI 0.04 to 0.19) |

  Selection picked a straight line for ap (held-out R^2 -0.03; flexible XGBoost -0.38,
  whose estimate is ~23% smaller, the same pull toward zero as on synthetic data) and
  cautious / flexible XGBoost for the residual (held-out R^2 only 0.03 / 0.08). Reading:
  (1) total effect about -39 m/day per 100 nT; on storm day 0 (mean ap 57 vs quiet 7 nT)
  that is about -20 m/day, in line with the event study. (2) Naive and DML agree on real
  data (robustness, as the plan says): both outcomes already subtract MSIS run with the
  actual F10.7, so little Sun-UV confounding is left, which is also why the confounders
  barely predict the residual. On synthetic data the UV error sat in the residual
  directly. (3) Model error has no same-interval link to ap, but is positive for ap 12-24
  h earlier (less decay than MSIS predicts the day after high ap): MSIS keeps the storm
  atmosphere too long, as in the B prototype. Reported as MSIS mistiming, not a reason
  to change the main lag; four lags were tried, so treat the 24 h result as supporting
  evidence until the refutations (with / without May and Oct 2024) are in.
- **Oct 31, DML refutations (Holla)**: `python -m src.dml_refute`, real table, 999 bootstrap
  draws, 20 placebo shifts. Equivalence band for the high-altitude control fixed before
  the run at +-50% of the main estimate (Sidhant; with 15 high objects the synthetic
  interval fits +-50% in 10/10 seeds, +-33% in 5/10). Effects in m/day per nT:

  | | `Y_total` (main -0.393) | `Y_residual` (main -0.053, CI covers 0) |
  |---|---|---|
  | High-altitude control (15 objects) | -0.023 (CI -0.026 to -0.021): inside +-0.196, PASS; ~6% of the main effect, not exactly 0 | -0.009 (CI -0.013 to -0.006); no band (main covers 0) |
  | Random common cause | moved -0.02 SE | moved +0.08 SE |
  | Shuffled storm timing (20) | mean -0.003, SD 0.031; none as large as the real | mean -0.007, SD 0.025; none as large |
  | Without May / Oct 2024 (storms 3, 4, 11) | -0.511 (CI -0.648 to -0.375) | -0.159 (CI -0.273 to -0.050) |
  | Lag 0 / 6 / 12 / 24 h without them | (not rerun) | -0.15 / -0.11 / 0.00 / +0.05 (12-24 h CIs cover 0) |

  Planted 12 h lag on synthetic data: the sweep peaks at 12 h (-0.0181 vs planted -0.0187;
  0 h -0.0109), so it can find a lag. Reading: the total effect is not carried by the
  superstorms (per nT it is larger without them: the response saturates at extreme ap).
  The 12-24 h positive model error (MSIS keeps the storm atmosphere too long) needs May /
  Oct 2024, so per the decision rule it is reported as driven by the extreme storms, not
  generalised. Without them, model error is negative in the storm interval itself: in
  ordinary storms objects fall faster than MSIS predicts; with them the two cancel.
- **Oct 31, model error by storm class (Holla)**: `python -m src.storm_class`, per-storm
  change (days 0..+3 minus days -10..-3) from `review_figures.storm_class_error`; t
  interval over storms for classes with >= 3 storms, none for severe (2) / extreme (1);
  exact permutation test of severe + extreme vs the rest over all C(20, 3) = 1,140
  splits. `Y_residual`: moderate +0.3 (CI -6.5 to 7.1), strong -4.2 (CI -19.3 to 10.8),
  severe +34.2 (Oct 2024) and -14.3 (Dec 31 2024), extreme +10.3 (May 2024); big minus rest
  +11.3 m/day, **p = 0.20**. `Y_total`: big minus rest -10.2, p = 0.30. So at storm level
  the "MSIS overshoots in extreme storms" headline is not supported: the overshoot is
  May and Oct 2024, and the other severe storm (Dec 31 2024) goes the other way. A clean
  baseline for the merged Oct storm (before its Oct 6 first dip) barely changes it
  (+38.9 / `Y_total` +5.2), so Oct 2024's positive total effect is not a baseline artefact.
  This agrees with the extreme-storm check (Sonu, Oct 31): Oct 2024 is a real overshoot,
  May 2024 a timing error, Dec 2024 an under-prediction, i.e. storm-specific errors and no
  general model error by storm size.
- **Oct 31, vulnerability map (Holla)**: `python -m src.vulnerability_map`, CausalForestDML
  (400 trees, min 50 rows per leaf), modifiers altitude / inclination / drag_sensitivity,
  controls F10.7, F10.7 81-day, season, local time; helper models by the DML rule;
  storm-blocked folds. Synthetic check (PLAN pass bar): per-object correlation with the
  planted effect 0.925, inclination importance 0.005 (planted none), average effect
  -0.0187 = planted, 5-seed map correlation > 0.9. Outcome for the real map is `Y_total`
  (question 2 asks who is hit hardest; PLAN did not name one), `Y_residual` second. Real
  `Y_total`: importance drag sensitivity 0.89, altitude 0.11, inclination 0.005; 5-seed map
  correlation >= 0.996. Hardest hit: high drag sensitivity at 600-700 km, about -2.5 to
  -3.4 m/day per nT (objects 4193, 13486, 233, 407, 13487), against about -0.05 for compact
  objects above 700 km; forest average -0.46 per nT (an unweighted row average, so not
  the same number as the DML estimate). Figure `results/vulnerability_map_Y_total.png`
  greys out cells with fewer than 3 objects nearby (111 of 189): the cohort has no object
  below 550 km with drag sensitivity above 0.08, so that corner would be extrapolation.
  Real `Y_residual` map is not stable (5-seed correlation down to 0.62, below the 0.9 bar)
  and is not interpreted. Caveat: the forest's own intervals treat rows as independent,
  so they are too narrow with 20 storms; read the map for its pattern, the effect size
  from DML.
- **Nov 7, model error by storm class in DML (Holla)**: `storm_class.dml_by_class`, one first stage on the whole main sample, final
  slope of the leftovers per class over the rows of that class's storms, storm-clustered
  wild bootstrap for >= 3 storms, exact permutation over the 1,140 storm splits for big vs
  rest. Per nT of ap: `Y_residual` moderate -0.140 (CI -0.260 to -0.013), strong -0.208
  (CI -0.341 to -0.072), severe -0.045, extreme +0.019 (no CI); big minus rest +0.191,
  p = 0.32. `Y_total` moderate -0.565 (CI -0.764 to -0.359), strong -0.591 (CI -0.719 to
  -0.457), severe -0.603, extreme -0.254; big minus rest +0.247, p = 0.45. Reading: in
  moderate and strong storms MSIS under-predicts the storm-time drag (objects lose more
  than predicted, CIs exclude 0); in the three biggest storms the per-nT model error is
  about 0 and the per-nT total effect smaller (saturation), but with three storms the
  difference to the rest is not significant at storm level. Synthetic check: the planted
  per-nT effect is the same in every class and DML-by-class recovers it in each (within
  25%), with no significant big-vs-rest gap.
- **Nov 7, vulnerability per unit drag sensitivity (Holla, Sonu's request)**: storm decay
  scales with area-to-mass by the drag equation, so the 0.89 drag importance of the `Y_total`
  map is mostly physics. `python -m src.vulnerability_map --per-drag`: the same forest on
  `Y_total / drag_sensitivity` ((m/day per nT) per m^2/kg). Real data: importance
  **altitude 0.993**, inclination 0.005, drag sensitivity 0.002; 5-seed map correlation
  >= 0.991. Within an altitude row the map is flat across drag sensitivity, i.e. the storm
  effect is proportional to drag sensitivity; per unit drag it falls from about -12 at
  617 km to about -4 at 747 km (about -28 at 519 km and -2 at 779 km, single supported
  cells), roughly an e-fold per 100-120 km, of the order of a thermospheric density scale
  height. Cross-check: the `Y_total` map divided cell by cell by its drag sensitivity
  correlates 0.96 with the per-drag forest over the supported cells. Synthetic check
  (planted per-drag effect depends on altitude only): altitude importance 0.88, drag
  0.09; per-object correlation with the planted per-drag effect 0.67 (both routes; the raw
  map gets 0.93), so this map is read for its altitude pattern, not cell values. Weighting
  rows by dt x drag^2 did not help (0.66). Figure `results/vulnerability_map_Y_total_per_drag.png`.
