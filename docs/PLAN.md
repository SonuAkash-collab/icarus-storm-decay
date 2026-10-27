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

**Not on `main` yet:** #7 (event-study checks) and #8 (offset ratio) were merged into
the `causal-event-study` branch, and #9 (high-altitude group, hourly ap) into
`causal-offset-ratio`, after #6 had already gone to `main`. They come to `main` in one
follow-up PR from C together with the review fixes below.

## A: Piyush (orbit data)

- [ ] **A2-A4 cohort and cleaning**: 400-800 km passive cohort, TLE cleaning, intervals
      that never bridge a `main_onset`. Most likely bottleneck for real data.
- [ ] **High-altitude control group**: 10-20 near-circular (ecc < 0.01) passive objects
      at 1,000-1,500 km (old rocket bodies, debris), kept as a separate group. Not
      LAGEOS itself (~5,900 km).
- [ ] **A7 precise-orbit check** (required): compare TLE-based decay with Swarm /
      GRACE-FO precise orbits.

## B: Sonu (space weather and physics)

- [ ] **Residual validation**
  - Outlier filter on residuals.
  - TLE-blurring check: does the post-storm over-prediction survive smoothing?
  - Baseline residual across all 20 storms; held-out quiet residual must be ~0.
  - Two estimands:
    (a) total storm effect = observed minus MSIS 2.1 run with ap replaced by the
        storm's pre-storm quiet ap (median ap over days -10..-3; ap is global, so one
        value per storm). F10.7 stays at its actual values, so the difference isolates
        the storm from Sun UV;
    (b) model error = observed minus MSIS 2.1 with actual ap (current `Y_residual`).
  - Report both, with and without the May 2024 and Oct 2024 storms.
  - Save (a) as its own column (proposed name `Y_total`, m/day): the offset ratio
    needs it for the storm loss.
- [ ] **B3 object catalog** (`src/objects.py`, branch `B3-objects`): CelesTrak SATCAT
      type, launch date, decay date; `is_active` (payloads with detected maneuvers in
      the year = active; debris, rocket bodies and payloads without maneuvers =
      passive); DISCOSweb area-to-mass where available. Also save `area_to_mass` and
      `drag_sensitivity_catalog = 2.2 * area_to_mass`. Output
      `data/processed/objects.parquet`. Tests + notebook 05.
- [ ] **B6 physics simulator** (`src/simulate.py`, branch `B6-simulator`): synthetic
      orbits decaying through pymsis MSIS 2.1 density with a storm boost of known size,
      TLE-style noise that grows during storms, output validated with
      `schema.validate`. Include a few objects above 1,000 km. Test: the physics
      residual pipeline recovers the planted boost.
- [ ] **B4-B5 at scale** (after A2 is merged): re-run `physics.py` for the whole
      cohort. Where the quiet-time fit of drag sensitivity is unusable (high-altitude
      group: predicted decay ~0, so observed / driver blows up), fall back to
      `drag_sensitivity_catalog` from B3, mark it in an extra column
      `drag_source` = "fit" / "catalog", and leave `Y_log_ratio` NaN.

## C: Siddhant Anil (event study and offset ratio, `causal-event-offset`)

- [x] **Synthetic additions** (#9, not on `main` yet): `n_high_alt` (1,000-1,500 km
      group) and `return_hourly` (hourly frame in the format of
      `omni_io.load_spaceweather()`, not in `table.attrs`).
- [x] **Event study** (#6, on `main`): stacked, clean controls, CR1 + wild cluster
      bootstrap by storm.
- [x] **Event-study checks** (#7, not on `main` yet): planted recovery, placebo-based
      pre-trend test, 500 placebo dates, with/without May/Oct 2024.
- [x] **Offset ratio** (#8, not on `main` yet): matched twins, station keeping removed
      via days -10..-3, ducking and recovery windows, debris placebo.
- [ ] **Follow-up PR to `main`** bringing #7-#9 in, with these fixes:
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
- [ ] **Sensitivity runs**: event study with `contam=(-3, 10)` as well as the default
      `"active"` (our headline is slow recovery, which can leak into the next storm's
      pre-period); offset ratio with a 5-day recovery window as well as 10 days.
- [ ] **Curved confounder option** in `make_synthetic`: UV model error that grows with
      altitude and saturates at high F10.7. Default stays linear so existing tests
      don't move.
- [ ] **Paper**: intro, data/physics setup, DAG, event study, offset ratio.

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

## Decision log

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
