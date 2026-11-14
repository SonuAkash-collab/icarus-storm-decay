# Sensitivity checks

Made by `python -m src.sensitivity` from the frozen analysis table and the cached physics drivers. Nothing frozen changes: variants are rebuilt in memory.

## 1. Hampel removal by timing

Passive main sample (ecc < 0.01, drag sensitivity, alt < 1000 km) before the `tle_ok` filter; share of rows with `hampel_outlier = 1`. Event day = whole days from `main_onset` of the linked storm (-10..+20 window).

| event day | rows | flagged | share % |
|---|---|---|---|
| days -10..-3 | 41371 | 1649 | 4.0 |
| days -2..-1 | 17145 | 618 | 3.6 |
| day 0 | 7896 | 1488 | 18.8 |
| days 1..2 | 17191 | 1431 | 8.3 |
| days 3..20 | 56647 | 2959 | 5.2 |
| outside storm windows | 48310 | 2308 | 4.8 |
| all rows | 188560 | 10453 | 5.5 |

By storm class (rows inside storm windows):

| storm_class | rows | flagged | share % |
|---|---|---|---|
| moderate | 47516 | 2280 | 4.8 |
| strong | 52587 | 3280 | 6.2 |
| severe | 28381 | 2068 | 7.3 |
| extreme | 11766 | 517 | 4.4 |
| all storm windows | 140250 | 8145 | 5.8 |

Class by event day (share %):

| storm class | days -10..-3 | days -2..-1 | day 0 | days 1..2 | days 3..20 |
|---|---|---|---|---|---|
| moderate | 3.2 | 3.4 | 9.2 | 6.1 | 6.2 |
| strong | 3.4 | 1.9 | 25.8 | 12.4 | 5.5 |
| severe | 7.0 | 9.6 | 43.7 | 8.7 | 5.3 |
| extreme | 4.5 | 2.2 | 37.2 | 3.1 | 3.4 |

## 2. Event study without the Hampel filter

Days 0-2 mean effect, 95% wild cluster bootstrap CI by storm (999 draws, seed 0, same settings as the freeze, no placebo dates). TLE endpoint checks kept in every variant.

Check, rebuilt k = 3.5 table against the saved table (largest absolute difference): Y_total 0, Y_residual 0, hampel_outlier 0, tle_ok 0, drag_sensitivity 0

| variant | outcome | effect | rows | storms_used |
|---|---|---|---|---|
| frozen (k = 3.5) | Y_total | -14.1 (-19.8 to -8.4) | 3566 | 12 |
| frozen (k = 3.5) | Y_residual | +0.2 (-3.2 to +3.2) | 3566 | 12 |
| put back (k = 3.5 flags off, B kept) | Y_total | -16.0 (-23.4 to -8.4) | 4730 | 12 |
| put back (k = 3.5 flags off, B kept) | Y_residual | -0.3 (-5.2 to +4.3) | 4730 | 12 |
| no filter (B refitted on all rows) | Y_total | -16.0 (-23.4 to -8.3) | 4730 | 12 |
| no filter (B refitted on all rows) | Y_residual | -0.3 (-5.3 to +4.3) | 4730 | 12 |
| k = 5 (B refitted) | Y_total | -15.2 (-21.4 to -8.7) | 3867 | 12 |
| k = 5 (B refitted) | Y_residual | -0.3 (-4.1 to +3.0) | 3867 | 12 |

- `put back`: the k = 3.5 flags are not used in the sample, B and predictions stay as frozen.
- `no filter`: no Hampel flags anywhere, so B is also refitted on every quiet row.
- `k = 5`: flags at 5 robust SDs (instead of 3.5), B refitted without them.

## 3. Q3 orbit-change screen

`offset_model.run` (model-error corrected, all storms, quiet-ap check as in the freeze), screen = largest net altitude change in a 21-day window that still counts as station keeping. 1 km is the frozen setting. Pairs = satellite-storm pairs.

| max_change_km | total | ducking | recovery | n_pairs | n_storms | n_objects |
|---|---|---|---|---|---|---|
| 0.5 | +1.18 (+0.98 to +1.36) | +1.16 (+1.05 to +1.28) | +0.01 (-0.20 to +0.21) | 262 | 20 | 21 |
| 1.0 | +1.17 (+0.97 to +1.36) | +1.16 (+1.05 to +1.27) | +0.01 (-0.22 to +0.21) | 278 | 20 | 21 |
| 2.0 | +1.18 (+1.00 to +1.36) | +1.17 (+1.06 to +1.29) | +0.01 (-0.21 to +0.21) | 286 | 20 | 21 |

## 4. Pairs in the main Q3 run

278 satellite-storm pairs (21 satellites, 20 storms), screen 1 km.

## Reading

- Hampel removal is concentrated at the storm onset: 18.8% of day-0 rows are flagged against 4.0% on days -10..-3 and 5.5% overall; the highest day-0 share is in severe storms (44%).
- put back (k = 3.5 flags off, B kept): Y_total -16.0 (-23.4 to -8.4), Y_residual -0.3 (-5.2 to +4.3) (frozen -14.1 (-19.8 to -8.4), +0.2 (-3.2 to +3.2)).
- no filter (B refitted on all rows): Y_total -16.0 (-23.4 to -8.3), Y_residual -0.3 (-5.3 to +4.3) (frozen -14.1 (-19.8 to -8.4), +0.2 (-3.2 to +3.2)).
- k = 5 (B refitted): Y_total -15.2 (-21.4 to -8.7), Y_residual -0.3 (-4.1 to +3.0) (frozen -14.1 (-19.8 to -8.4), +0.2 (-3.2 to +3.2)).
- Q3 offset at 0.5 / 1 / 2 km: 1.18 / 1.17 / 1.18; the screen only changes the pair count (262 / 278 / 286).

## Summary table

Change % = (variant - frozen) / |frozen|. Where the frozen value is under 20% of its parent headline (Y_residual vs Y_total, the Q3 recovery part vs the total) the ratio explodes, so the change is measured against the parent headline instead (column `measured against`). Flag: |change| > 20%.

**Flags:** no headline moves by more than 20%.

| headline | variant | frozen | variant value | change % | measured against | CI includes 0 | flag |
|---|---|---|---|---|---|---|---|
| Y_total days 0-2 (m/day) | put back (k = 3.5 flags off, B kept) | -14.1 (-19.8 to -8.4) | -16.0 (-23.4 to -8.4) | -13.8 | own baseline | False -> False |  |
| Y_residual days 0-2 (m/day) | put back (k = 3.5 flags off, B kept) | +0.2 (-3.2 to +3.2) | -0.3 (-5.2 to +4.3) | -3.1 | parent headline | True -> True |  |
| Y_total days 0-2 (m/day) | no filter (B refitted on all rows) | -14.1 (-19.8 to -8.4) | -16.0 (-23.4 to -8.3) | -13.8 | own baseline | False -> False |  |
| Y_residual days 0-2 (m/day) | no filter (B refitted on all rows) | +0.2 (-3.2 to +3.2) | -0.3 (-5.3 to +4.3) | -3.1 | parent headline | True -> True |  |
| Y_total days 0-2 (m/day) | k = 5 (B refitted) | -14.1 (-19.8 to -8.4) | -15.2 (-21.4 to -8.7) | -7.7 | own baseline | False -> False |  |
| Y_residual days 0-2 (m/day) | k = 5 (B refitted) | +0.2 (-3.2 to +3.2) | -0.3 (-4.1 to +3.0) | -3.4 | parent headline | True -> True |  |
| Q3 corrected total offset | screen 0.5 km | +1.17 (+0.97 to +1.36) | +1.18 (+0.98 to +1.36) | +0.5 | own baseline | False -> False |  |
| Q3 ducking part | screen 0.5 km | +1.16 (+1.05 to +1.27) | +1.16 (+1.05 to +1.28) | +0.1 | own baseline | False -> False |  |
| Q3 recovery part | screen 0.5 km | +0.01 (-0.22 to +0.21) | +0.01 (-0.20 to +0.21) | +0.4 | parent headline | True -> True |  |
| Q3 corrected total offset | screen 2 km | +1.17 (+0.97 to +1.36) | +1.18 (+1.00 to +1.36) | +0.8 | own baseline | False -> False |  |
| Q3 ducking part | screen 2 km | +1.16 (+1.05 to +1.27) | +1.17 (+1.06 to +1.29) | +0.8 | own baseline | False -> False |  |
| Q3 recovery part | screen 2 km | +0.01 (-0.22 to +0.21) | +0.01 (-0.21 to +0.21) | +0.0 | parent headline | True -> True |  |
