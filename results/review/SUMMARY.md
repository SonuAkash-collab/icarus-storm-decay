# Review figure pack

Made by `python -m src.review_figures` from the analysis table and results/causal. One line per figure with its key number.

1. `fig1_may2024_cohort.png`: the median decay rate of 250 passive objects goes from -18 m/day (3 days before main_onset) to -60 m/day (first day after), 3.4x.
2. `fig2_storm_catalog.png`: 20 storms with Dst < -50 nT (Apr 2024 - Apr 2025): 11 moderate, 6 strong, 2 severe, 1 extreme; deepest Dst -406 nT (May 2024).
3. `fig3_physics_check.png`: held-out quiet residual of MSIS 2.1 on 250 objects -0.13 m/day (SE 1.07), i.e. zero within noise; example objects NORAD 5118 and 10974.
4. `fig4_event_study.png`: days 0-2 effect, total storm effect -14.1 m/day (95% CI -19.8 to -8.4); model error +0.2 m/day (95% CI -3.2 to +3.2). Day -1 is the reference.
5. `fig5_model_error_by_class.png`: mean Y_residual change (days 0..+3 minus days -10..-3) by class: moderate +0.3 (11), strong -4.2 (6), severe +10.0 (2), extreme +10.3 (1) m/day (storms in brackets). Extreme (one storm, CI over objects) +10.3 [+8.1, +12.4]; severe (two storms, CI over objects) +10.0 [+5.5, +14.4], but the two severe storms disagree (+34.2 and -14.3). Positive = MSIS predicts more decay than observed; caveat from notebook 04: the May 2024 value vanishes with 3-day smoothing (TLE blur).
6. `fig6_validation.png`: GRACE-FO-1 Oct 2024 TLE vs precise decay rate r = 0.70 (Piyush's A7 result; the left panel is cut from his saved figure); fitted vs catalog drag of rocket bodies Spearman rho = 0.81 (n = 116).
7. `fig7_operator_offset.png`: Starlink total offset 1.19 (95% CI 0.97 to 1.42), of which ducking part 1.07 (95% CI 0.97 to 1.18) and recovery part 0.12 (95% CI -0.10 to 0.33); observed loss is smaller than predicted in 17 of 20 storms.
