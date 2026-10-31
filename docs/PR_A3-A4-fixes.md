# A3-A4 fixes

## What changed

- Added regression tests for duplicate TLE removal, a planted inclination jump,
  passive climbs, five-day gap removal, storm-onset boundaries, altitude-band
  selection, and precise-orbit parsing.
- Replaced the zero-kilometre passive-climb rule with a per-object robust TLE-noise
  estimate and a three-times-noise tolerance.
- Omitted intervals with gaps over five days and kept the onset-boundary exclusion.
- Rebuilt the cohort across 100 km altitude bands: 39/63/82/66 objects in the
  400-800 km bands, plus 3 objects in each 100 km band from 1,000-1,500 km.
- Added the before/after cleaning and May 2024 checks in
  `notebooks/05_orbit_cleaning_checks.ipynb`.
- Added `src/precise_orbit.py` and an A7 GRACE-FO-1 comparison using the public
  TU Graz reduced-dynamic orbit archive.

## Key numbers

- 203,954 deduplicated TLEs and 198,388 intervals.
- 7 intervals omitted for gaps over five days; 5,294 omitted for crossing a
  storm main onset; 416 intervals flagged for passive climb.
- Tight point-level diagnostic: 15 flagged TLEs (0.0074%): 9 during storm
  intervals, 5 in post-event windows, and 1 on calm time.
- A7 May 2024 GRACE-FO-1 comparison: 19 aligned days, centred correlation
  0.907, centred RMSE 0.097 km, and slope difference 0.0025 km/day.

## Verification

`pytest tests/test_orbit.py -q` passes 9 tests.
