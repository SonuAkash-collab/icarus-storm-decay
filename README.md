# ICARUS: What Does a Solar Storm Cost an Orbit?

Causal effects of geomagnetic storms on satellite orbital decay, and how much of the damage operators undo.
Advanced Data Analytics course project, PES University.

## Team
- Siddanth Anil (PES1UG24AM274)
- Sonu Akash Soundararajan (PES1UG24AM908)
- Piyush Shiv (PES1UG24AM191)
- Sidhant Holla (PES1UG24AM466)

## Setup
    python -m venv .venv
    .venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
    pip install -r requirements.txt

## Folders
- `data/` raw and cleaned data (not in git; shared via Drive)
- `src/` reusable code (cleaning, features, analysis)
- `notebooks/` exploration, numbered 01_, 02_, ...
- `results/` final plots and tables
- `docs/PLAN.md` task list, rules and decision log

## Reproduce
The course deliverable is `notebooks/ICARUS_final.ipynb`. Run every command from the repository root, in this order.

1. **Download** (needs a free Space-Track account and a DISCOSweb token in a local `.env`). You can skip this step if
   you have the shared `data/` folder from Drive.
   - Space weather and the storm catalogue (OMNI2, no account):

         python -c "from src.omni_io import build_spaceweather; from src.storms import storm_catalog; from src.physics import load_msis_weather; sw = build_spaceweather(); storm_catalog(sw.set_index('time')['dst'], out_path='default'); load_msis_weather(force=True)"

   - Passive cohort TLEs and decay intervals (`spacetrack_io.select_passive_cohort`, `download_tle_history`,
     `orbit.clean_tle_history`, `orbit.build_decay_intervals`; see `notebooks/05_orbit_cleaning_checks.ipynb`) ->
     `data/raw/piyush/processed/cohort_tles_cleaned.parquet` and `data/processed/orbit_intervals_a2_a4.parquet`.
   - Active satellites: `python -m src.active_cohort`
   - Object catalogue (SATCAT, DISCOSweb): `objects.build_objects` in `notebooks/06_objects.ipynb` -> `data/processed/objects.parquet`.
2. **Build the analysis table** (~40 min the first time, ~20 s from the physics cache):

       python -m src.build_table

3. **Runs** (results go to the git-ignored `results/causal/` and `results/dml/`; several hours in total):

       python -m src.causal_run
       python -m src.dml_run
       python -m src.dml_refute
       python -m src.storm_class --outcome Y_residual
       python -m src.storm_class --outcome Y_total
       python -m src.vulnerability_map
       python -m src.vulnerability_map --per-drag
       python -m src.extreme_check
       python -m src.sensitivity

4. **Figures** (the nine review figures and `results/review/SUMMARY.md`):

       python -m src.review_figures

5. **Notebook** (about 4 minutes with `FAST = True`, which loads the saved results of steps 2-3; set `FAST = False`
   in its first cell to recompute them instead). The last section checks every headline number against `SUMMARY.md`.

       python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/ICARUS_final.ipynb

## How we work
Never push to `main`. Make a branch per task, open a pull request, get one review, merge.
