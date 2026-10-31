# Review figure pack

Made by `python -m src.review_figures` from the analysis table and results/causal. One line per figure with its key number.

## Headline

- Storms cause extra decay beyond what a quiet-ap atmosphere would give: total storm effect, days 0-2, -14.1 m/day (95% CI -19.8 to -8.4).
- MSIS 2.1 has no general post-storm error: event-study model error +0.2 m/day (95% CI -3.2 to +3.2); days 0..+3 change across all 20 storms +0.8 m/day (95% CI -5.8 to +7.4), -1.8 m/day (95% CI -7.6 to +4.0) without May/Oct 2024.
- Extreme storms (250 objects, fig 8), verdict: **mixed (May 2024: survives smoothing, not in calm-to-calm; Oct 2024: real model overshoot)**.
  - Oct 2024 (severe): a real model overshoot. The days 0..+3 over-prediction survives 3-day smoothing (+28.4 m/day (95% CI +22.8 to +33.9)) and MSIS also predicts more altitude loss than observed from a calm day before to a calm day after: observed / predicted 0.86 (95% CI 0.84 to 0.87), +169 m (95% CI +144 to +197) after removing the model's local quiet error, in every altitude band (ratio 0.85-0.90).
  - May 2024 (extreme): a timing error, not an overshoot of the total. The days 0..+3 excess survives smoothing (+6.9 m/day (95% CI +4.6 to +9.2)), but the calm-to-calm loss is not over-predicted once the model's own error is removed: -1 m (95% CI -10 to +7) against its +-30-day quiet error, -75 m (95% CI -92 to -59) against its pre-storm error (raw observed / predicted 0.91; MSIS already over-predicted the decay before the storm).
  - Dec 2024 (severe): the opposite, MSIS under-predicts the loss (observed / predicted 1.19 (95% CI 1.17 to 1.22)).
- So the headline is not "models recover too slowly after storms": MSIS 2.1 errors in the largest storms are storm-specific, and only Oct 2024 shows a real overshoot.

## Figures

1. `fig1_may2024_cohort.png`: the median decay rate of 250 passive objects goes from -18 m/day (3 days before main_onset) to -60 m/day (first day after), 3.4x.
2. `fig2_storm_catalog.png`: 20 storms with Dst < -50 nT (Apr 2024 - Apr 2025): 11 moderate, 6 strong, 2 severe, 1 extreme; deepest Dst -406 nT (May 2024).
3. `fig3_physics_check.png`: held-out quiet residual of MSIS 2.1 on 250 objects -0.13 m/day (SE 1.07), i.e. zero within noise; example objects NORAD 5118 and 10974.
4. `fig4_event_study.png`: days 0-2 effect, total storm effect -14.1 m/day (95% CI -19.8 to -8.4); model error +0.2 m/day (95% CI -3.2 to +3.2). Day -1 is the reference.
5. `fig5_model_error_by_class.png`: mean Y_residual change (days 0..+3 minus days -10..-3) by class: moderate +0.3 (11), strong -4.2 (6), severe +10.0 (2), extreme +10.3 (1) m/day (storms in brackets). Extreme (one storm, CI over objects) +10.3 [+8.1, +12.4]; severe (two storms, CI over objects) +10.0 [+5.5, +14.4], but the two severe storms disagree (+34.2 and -14.3). Positive = MSIS predicts more decay than observed. Caveat (fig 8): on the full cohort the May 2024 value does not vanish with 3-day smoothing (+6.9 m/day), but it is a timing error, not an overshoot of the storm's total loss; of the two severe storms only Oct 2024 is a real overshoot, Dec 2024 is an under-prediction.
6. `fig6_validation.png`: GRACE-FO-1 Oct 2024 TLE vs precise decay rate r = 0.70 (Piyush's A7 result; the left panel is cut from his saved figure); fitted vs catalog drag of rocket bodies Spearman rho = 0.81 (n = 116).
7. `fig7_operator_offset.png`: Starlink total offset 1.19 (95% CI 0.97 to 1.42), of which ducking part 1.07 (95% CI 0.97 to 1.18) and recovery part 0.12 (95% CI -0.10 to 0.33); observed loss is smaller than predicted in 17 of 20 storms.
8. `fig8_extreme_check.png`: does the extreme-storm over-prediction survive on 250 passive objects? (A) Days 0..+3 residual change, MSIS 2.1 raw -> smoothed over 3 days: May 2024 +14.8 -> +6.9, Oct 2024 +34.6 -> +28.4, Dec 2024 -15.0 -> -17.3 m/day (all 20 storms +0.8 -> -1.1); best lag -3 h, correlation only 0.003 above 0 h. (B) Running observed - predicted loss from the calm day before (dashed: the pre-storm error carried on): May keeps rising at its pre-storm rate and gives part back after day 6, Oct jumps after the storm and stays up, Dec falls. (C) Calm-to-calm loss, last calm day in days -10..-3 to the first calm day after recovery_end (calm = every 3-h ap < 27; 13 of 20 storms have both): observed / predicted May 0.91, Oct 0.86, Dec 1.19; minus the +-30-day quiet error May -1 m, Oct +169 m, Dec -101 m. With the calm day after pushed later, observed / predicted May/Oct/Dec +0 d 0.91/0.86/1.19; +2 d 0.91/0.86/1.18; +3 d 0.91/0.87/1.17. High-altitude control (1,040-1,211 km): |difference| at most 7 m. Verdict (PLAN.md rules): mixed (May 2024: survives smoothing, not in calm-to-calm; Oct 2024: real model overshoot).
