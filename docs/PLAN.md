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

- [ ] **Synthetic additions** (`src/synthetic.py`):
  - option for a high-altitude group (1,000-1,500 km) in `make_objects` / `make_synthetic`;
  - option to also return the hourly space weather as a frame in the same format as
    `omni_io.load_spaceweather()` (time index; `ap`, `f107`, `f107_81d`), so lag code
    runs the same on real and synthetic data. Not in `table.attrs`: `save_table`
    writes attrs to Parquet and a Series there can break saving.
- [ ] **Event study** (`src/event_study.py`): stacked, with clean controls (objects or
      days not inside any other storm's -10..+20 window); drop contaminated windows and
      report how many. Checks: recovers the planted effect, flat pre-trends, 500
      placebo storm dates centre on zero, with and without May/Oct 2024.
- [ ] **Offset ratio** (`src/offset_ratio.py`): actives matched to passive twins on
      altitude, inclination and `drag_sensitivity`; subtract each active satellite's
      own station-keeping rate on days -10..-3. Windows: ducking (onset to
      recovery end) and recovery (next ~10 days). Debris-vs-debris placebo (offset ~0).
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
  - ap lags 0 / 6 / 12 / 24 h: main lag fixed in advance, all four reported.
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
