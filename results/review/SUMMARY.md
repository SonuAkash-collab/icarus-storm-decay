# Review figure pack

Made by `python -m src.review_figures` from the analysis table, results/causal and results/dml. One line per figure with its key number.

## Headline

### Q1. Extra decay from storms and model error

- Storms cause extra decay beyond what a quiet-ap atmosphere would give: total storm effect, days 0-2, -14.1 m/day (95% CI -19.8 to -8.4) (event study); DML -0.393 (95% CI -0.527 to -0.268) m/day per nT of interval-mean ap, i.e. -39 m/day per 100 nT.
- In ordinary storms MSIS gets the storm-window total about right but under-predicts decay at peak ap (DML by class: moderate (11 storms) -0.133 (95% CI -0.247 to -0.010) and strong (6 storms) -0.210 (95% CI -0.348 to -0.068) m/day per nT, CIs exclude 0); in severe storms its errors are storm-specific (Oct 2024 overshoot, May 2024 timing, Dec 2024 under-prediction). Storm-window total: event-study model error days 0-2 +0.2 m/day (95% CI -3.2 to +3.2); days 0..+3 change across all 20 storms +0.8 m/day (95% CI -5.8 to +7.4), -1.8 m/day (95% CI -7.6 to +4.0) without May/Oct 2024.
- Extreme storms (250 objects, fig 8), verdict: **mixed (May 2024: survives smoothing, not in calm-to-calm; Oct 2024: real model overshoot)**.
  - Oct 2024 (severe): a real model overshoot. The days 0..+3 over-prediction survives 3-day smoothing (+28.4 m/day (95% CI +22.8 to +33.9)) and MSIS also predicts more altitude loss than observed from a calm day before to a calm day after: observed / predicted 0.86 (95% CI 0.84 to 0.87), +169 m (95% CI +144 to +197) after removing the model's local quiet error, in every altitude band (ratio 0.85-0.90).
  - May 2024 (extreme): a timing error, not an overshoot of the total. The days 0..+3 excess survives smoothing (+6.9 m/day (95% CI +4.6 to +9.2)), but the calm-to-calm loss is not over-predicted once the model's own error is removed: -1 m (95% CI -10 to +7) against its +-30-day quiet error, -75 m (95% CI -92 to -59) against its pre-storm error (raw observed / predicted 0.91; MSIS already over-predicted the decay before the storm).
  - Dec 2024 (severe): the opposite, MSIS under-predicts the loss (observed / predicted 1.19 (95% CI 1.17 to 1.22)).
- The 12-24 h lag effect in the model error (less decay than MSIS predicts the day after high ap, i.e. MSIS keeps the storm atmosphere too long): all storms 12 h +0.085 (95% CI -0.004 to +0.179), 24 h +0.117 (95% CI +0.035 to +0.200); without May + Oct 2024 (storms 3,4,11) 12 h +0.002 (95% CI -0.121 to +0.121), 24 h +0.048 (95% CI -0.055 to +0.152); so it only appears with May + Oct 2024 and is not a general result.

### Q2. Who is hit hardest

- Storm vulnerability = drag sensitivity x an altitude factor; per unit drag the effect falls about 3x from ~620 to ~750 km (-10.8 to -3.7 (m/day per nT) per m^2/kg, mean of the supported cells of each altitude row; fig 9).

### Q3. Operator offset (Starlink)

- Starlinks offset essentially all of the predicted storm loss (model-error corrected 1.17, 95% CI 0.97-1.36; 21 satellites, 20 storms), almost all of it during the storm (ducking part 1.16 (95% CI 1.05 to 1.27), recovery part +0.01, 95% CI -0.22 to +0.21, i.e. ~0). Without the 6 storms whose pre-storm quiet ap was disturbed (14 storms left): 1.15 (95% CI 1.04 to 1.26). Without May / Oct 2024: 1.25 (95% CI 1.08 to 1.41).

## Refutations (DML, m/day per nT)

- `Y_total` (main -0.393 (95% CI -0.527 to -0.268)):
  - High-altitude control (15 objects): -0.023 (95% CI -0.026 to -0.021); equivalence band +-0.197: PASS.
  - Random common cause: estimate moves -0.01 SE.
  - Shuffled storm timing (100 shifts): mean +0.004, SD 0.028; placebo p = 0.010 (0 of 100 as large as the real estimate).
  - Without May + Oct 2024 (storms 3,4,11): -0.503 (95% CI -0.642 to -0.361).
- `Y_residual` (main -0.050 (95% CI -0.141 to +0.047)):
  - High-altitude control (15 objects): -0.009 (95% CI -0.013 to -0.006); no equivalence test (main estimate covers 0).
  - Random common cause: estimate moves -0.00 SE.
  - Shuffled storm timing (100 shifts): mean +0.004, SD 0.025; placebo p = 0.059 (5 of 100 as large as the real estimate).
  - Without May + Oct 2024 (storms 3,4,11): -0.152 (95% CI -0.267 to -0.042).
  - Lag sweep without May + Oct 2024 (`Y_residual`): 0 h -0.155 (95% CI -0.266 to -0.051), 6 h -0.108 (95% CI -0.232 to +0.011), 12 h +0.002 (95% CI -0.121 to +0.121), 24 h +0.048 (95% CI -0.055 to +0.152).
- Model error by storm class in DML (`Y_residual`): moderate -0.133 (95% CI -0.247 to -0.010), strong -0.210 (95% CI -0.348 to -0.068), severe -0.037 (no CI, 2 storm(s)), extreme +0.021 (no CI, 1 storm(s)); severe + extreme vs the rest, exact permutation p = 0.32.

## Question 2: vulnerability map (fig 9)

- `Y_total` forest (250 objects): importance drag_sensitivity 0.892, alt_km 0.101, inc_deg 0.007; 5-seed map correlation min 0.986 (mean 0.994); forest average -0.453 m/day per nT (unweighted row average, not the DML estimate).
- Per unit drag sensitivity (`Y_total_per_drag`): importance alt_km 0.993, inc_deg 0.005, drag_sensitivity 0.002; 5-seed map correlation min 0.995 (mean 0.996). Once size is taken out, altitude carries the map: the storm effect is proportional to drag sensitivity.
- Per-drag altitude pattern: -10.8 at 617 km to -3.7 at 747 km ((m/day per nT) per m^2/kg), about 2.9x; read for the pattern, not single cells. Cross-check (Y_total map / drag sensitivity vs the per-drag forest, supported cells): correlation 0.96.

## Figures

1. `fig1_may2024_cohort.png`: the median decay rate of 250 passive objects goes from -18 m/day (3 days before main_onset) to -60 m/day (first day after), 3.4x.
2. `fig2_storm_catalog.png`: 20 storms with Dst < -50 nT (Apr 2024 - Apr 2025): 11 moderate, 6 strong, 2 severe, 1 extreme; deepest Dst -406 nT (May 2024).
3. `fig3_physics_check.png`: held-out quiet residual of MSIS 2.1 on 250 objects -0.13 m/day (SE 1.07), i.e. zero within noise; example objects NORAD 5118 and 10974.
4. `fig4_event_study.png`: days 0-2 effect, total storm effect -14.1 m/day (95% CI -19.8 to -8.4); model error +0.2 m/day (95% CI -3.2 to +3.2). Day -1 is the reference.
5. `fig5_model_error_by_class.png`: mean Y_residual change (days 0..+3 minus days -10..-3) by class: moderate +0.3 (11), strong -4.2 (6), severe +10.0 (2), extreme +10.3 (1) m/day (storms in brackets). Extreme (one storm, CI over objects) +10.3 [+8.1, +12.4]; severe (two storms, CI over objects) +10.0 [+5.5, +14.4], but the two severe storms disagree (+34.2 and -14.3). Positive = MSIS predicts more decay than observed. Caveat (fig 8): on the full cohort the May 2024 value does not vanish with 3-day smoothing (+6.9 m/day), but it is a timing error, not an overshoot of the storm's total loss; of the two severe storms only Oct 2024 is a real overshoot, Dec 2024 is an under-prediction.
6. `fig6_validation.png`: GRACE-FO-1 Oct 2024 TLE vs precise decay rate r = 0.70 (Piyush's A7 result; the left panel is cut from his saved figure); fitted vs catalog drag of rocket bodies Spearman rho = 0.81 (n = 116).
7. `fig7_operator_offset.png`: Starlink total offset 1.19 (95% CI 0.77 to 1.64), of which ducking part 1.23 (95% CI 1.11 to 1.36) and recovery part -0.04 (95% CI -0.47 to 0.38); observed loss is smaller than predicted in 16 of 20 storms.
8. `fig8_extreme_check.png`: does the extreme-storm over-prediction survive on 250 passive objects? (A) Days 0..+3 residual change, MSIS 2.1 raw -> smoothed over 3 days: May 2024 +14.8 -> +6.9, Oct 2024 +34.6 -> +28.4, Dec 2024 -15.0 -> -17.3 m/day (all 20 storms +0.8 -> -1.1); best lag -3 h, correlation only 0.003 above 0 h. (B) Running observed - predicted loss from the calm day before (dashed: the pre-storm error carried on): May keeps rising at its pre-storm rate and gives part back after day 6, Oct jumps after the storm and stays up, Dec falls. (C) Calm-to-calm loss, last calm day in days -10..-3 to the first calm day after recovery_end (calm = every 3-h ap < 27; 13 of 20 storms have both): observed / predicted May 0.91, Oct 0.86, Dec 1.19; minus the +-30-day quiet error May -1 m, Oct +169 m, Dec -101 m. With the calm day after pushed later, observed / predicted May/Oct/Dec +0 d 0.91/0.86/1.19; +2 d 0.91/0.86/1.18; +3 d 0.91/0.87/1.17. High-altitude control (1,040-1,211 km): |difference| at most 7 m. Verdict (PLAN.md rules): mixed (May 2024: survives smoothing, not in calm-to-calm; Oct 2024: real model overshoot).
9. `fig9_vulnerability.png`: (A) per-object storm effect from the `Y_total` causal forest, (B) the `Y_total` map and (C) the map per unit drag sensitivity, both at the median inclination: the effect grows with drag sensitivity and, per unit drag, falls from -10.8 at 617 km to -3.7 at 747 km.
