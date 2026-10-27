"""Double machine learning (DML) estimate of the storm effect on residual decay.

Question: if ap goes up by 1 nT, how much does Y_residual (m/day) change for the
same Sun, season and orbit? Negative = more decay than the physics model predicts.

Why not a plain regression: Sun UV (f107) raises both storm activity and the
physics-model error, so a plain slope of Y_residual on ap credits the UV part to
the storm (confounding). DML removes the confounders' influence from both sides:

    Y_res = Y - E[Y | X]      (model_y, XGBoost)
    T_res = T - E[T | X]      (model_t, XGBoost)
    theta = slope of Y_res on T_res (weighted by dt_days, no intercept)

This is the partially linear model Y = theta * T + g(X) + e, with X the adjustment
set from src/dag.py (schema.CONFOUNDERS, season as sin/cos of day_of_year).

Treatment: mean ap over the interval, T_ap_int / dt_days (nT). Y_residual is a rate
(m/day) and storm intervals are shorter than quiet ones, so the interval total
T_ap_int would mix interval length into the effect. synthetic.py plants the effect
per mean ap in the same way.

Cross-fitting: each row's E[Y|X] and E[T|X] come from models trained on other
folds. All objects see the same storm on the same days, so folds are whole time
blocks (cluster: storm_id, or the calendar week for rows outside every event
window). With row-wise folds the models would learn each storm's dates from the
other objects on the same day, T_res would be ~0 and the effect would vanish.

Nuisance models: for each of E[Y|X] and E[T|X] we score three candidates (straight
line, cautious XGBoost, flexible XGBoost) on the same time-block folds and keep the
one with the lowest held-out error. The choice uses only observed data, so the same
rule works on the real table. On synthetic data ap is close to unpredictable from X
on unseen storms; a flexible model then fits noise, and that noise in T_res pulls
theta toward zero (about -8% over 50 seeds), so letting the data choose matters.

Inference: ~20 storms, so errors are clustered by the same time blocks: CR1 and a
wild cluster bootstrap with Webb weights, as in src/event_study.py.

When the effect differs by object (it does: altitude, drag sensitivity), theta is a
weighted average of the per-object effects, weights = dt_days * T_res^2.
planted_theta computes that target from the synthetic truth.
"""

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.dag import check_adjustment
from src.schema import CONFOUNDERS, main_sample

TREATMENT = "ap_mean"
# CONFOUNDERS with day_of_year as a season wave (so late Dec and early Jan are close)
FEATURES = ["f107", "f107_81d", "season_sin", "season_cos", "alt_km", "inc_deg",
            "local_solar_time", "drag_sensitivity"]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def time_clusters(df):
    """Cluster label per row: 's<storm_id>' inside an event window, else 'w<year-week>'."""
    week = df["t_mid"].dt.strftime("%G-%V")
    return np.where(df["storm_id"].notna(), "s" + df["storm_id"].astype(str), "w" + week)


def prepare(table, outcome="Y_residual", use_main_sample=True):
    """Rows and columns DML needs: main sample, outcome present, treatment, features, cluster."""
    check_adjustment(CONFOUNDERS)
    df = main_sample(table) if use_main_sample else table
    df = df[df[outcome].notna()]
    angle = 2 * np.pi * df["day_of_year"] / 365.25
    return df.assign(ap_mean=df["T_ap_int"] / df["dt_days"],
                     season_sin=np.sin(angle), season_cos=np.cos(angle),
                     cluster=time_clusters(df))


def time_folds(clusters, n_folds=5, seed=0):
    """Cross-fitting splits that keep every cluster (storm / week) inside one fold."""
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(gkf.split(np.zeros(len(clusters)), groups=clusters))


def candidate_models(seed=0):
    """Nuisance model candidates, from simplest to most flexible."""
    def xgb(n, depth, min_child_weight):
        return XGBRegressor(n_estimators=n, max_depth=depth, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=min_child_weight,
                            n_jobs=-1, random_state=seed)
    return {"linear": LinearRegression(),
            "xgb_cautious": xgb(100, 3, 500),
            "xgb": xgb(300, 4, 1)}


def select_model(X, y, w, folds, seed=0):
    """Pick the candidate with the lowest held-out weighted MSE on the given folds.

    Returns (name, scores) with scores[name] = {"mse", "r2"} (held-out R^2; below 0
    means worse than predicting the mean).
    """
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    var = np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    scores = {}
    for name, model in candidate_models(seed).items():
        pred = np.empty(len(y))
        for train, test in folds:
            fitted = clone(model).fit(X[train], y[train], sample_weight=w[train])
            pred[test] = fitted.predict(X[test])
        mse = np.average((y - pred) ** 2, weights=w)
        scores[name] = {"mse": mse, "r2": 1 - mse / var}
    return min(scores, key=lambda n: scores[n]["mse"]), scores


def wls_cluster(X, y, w, clusters, n_boot=999, seed=0):
    """WLS of y on the columns of X; CR1 and wild cluster bootstrap (Webb) by cluster.

    Returns a dict for the first column of X: coef, se_cr1, se_boot, ci_lo, ci_hi
    (95% basic bootstrap interval), n_clusters. n_boot=0 skips the bootstrap.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    sw = np.sqrt(np.asarray(w, float))
    A = X * sw[:, None]
    bread = np.linalg.pinv(A.T @ A)
    P = bread @ A.T                       # beta = P @ (sqrt(w) * y)
    beta = P @ (y * sw)
    resid = y * sw - A @ beta

    uniq, cl_idx = np.unique(clusters, return_inverse=True)
    G, n, p = len(uniq), len(y), X.shape[1]
    scores = np.zeros((G, p))
    np.add.at(scores, cl_idx, A * resid[:, None])
    cr1 = G / (G - 1) * (n - 1) / (n - p) * bread @ (scores.T @ scores) @ bread

    out = {"coef": beta[0], "se_cr1": np.sqrt(max(cr1[0, 0], 0)),
           "se_boot": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_clusters": G}
    if n_boot > 1:
        # beta* = beta + P @ (resid * v): sum the residual part per cluster once,
        # then each draw is a weighted sum over clusters
        per_cluster = np.zeros(G)
        np.add.at(per_cluster, cl_idx, P[0] * resid)
        v = np.random.default_rng(seed).choice(WEBB, size=(n_boot, G))
        draws = beta[0] + v @ per_cluster
        lo_q, hi_q = np.percentile(draws - beta[0], [97.5, 2.5])
        out.update(se_boot=draws.std(ddof=1), ci_lo=beta[0] - lo_q, ci_hi=beta[0] - hi_q)
    return out


def naive(df, outcome="Y_residual", n_boot=999, seed=0):
    """Plain slope of the outcome on mean ap (intercept only, no confounders)."""
    X = np.column_stack([df[TREATMENT], np.ones(len(df))])
    return wls_cluster(X, df[outcome], df["dt_days"], df["cluster"], n_boot, seed)


def linear(df, outcome="Y_residual", n_boot=999, seed=0):
    """Slope on mean ap with the confounders as straight-line controls (ordinary regression)."""
    X = np.column_stack([df[TREATMENT], df[FEATURES], np.ones(len(df))])
    return wls_cluster(X, df[outcome], df["dt_days"], df["cluster"], n_boot, seed)


def dml(df, outcome="Y_residual", n_folds=5, n_boot=999, seed=0, model_y=None, model_t=None):
    """DML with selected nuisance models and time-block cross-fitting.

    model_y / model_t: candidate names to force a helper model; None = select by
    held-out error. Returns the wls_cluster dict plus model_y / model_t (names used),
    selection (held-out scores of every candidate, empty if forced), the cross-fitted
    residuals (y_res, t_res) and the econml point estimate (theta_econml, should
    equal coef).
    """
    folds = time_folds(df["cluster"].to_numpy(), n_folds, seed)
    W, w = df[FEATURES].to_numpy(), df["dt_days"].to_numpy()
    scores_y = scores_t = {}
    if model_y is None:
        model_y, scores_y = select_model(W, df[outcome], w, folds, seed)
    if model_t is None:
        model_t, scores_t = select_model(W, df[TREATMENT], w, folds, seed)
    models = candidate_models(seed)
    est = LinearDML(model_y=models[model_y], model_t=models[model_t], cv=folds, random_state=seed)
    est.fit(df[outcome].to_numpy(), df[TREATMENT].to_numpy(), W=W, sample_weight=w, cache_values=True)
    y_res, t_res = (np.ravel(r) for r in est.residuals_[:2])
    out = wls_cluster(t_res[:, None], y_res, w, df["cluster"], n_boot, seed)
    out.update(model_y=model_y, model_t=model_t, selection={"y": scores_y, "t": scores_t},
               y_res=y_res, t_res=t_res, theta_econml=float(np.ravel(est.const_marginal_effect())[0]))
    return out


def planted_theta(df, t_res):
    """Synthetic data only: the weighted-average planted effect DML targets (m/day per nT).

    true_effect = -true_tau * (ap_mean - AP_QUIET), so the per-row slope is -true_tau;
    DML averages it with weights dt_days * t_res^2. Pass the true leftover ap
    (true_leftover_ap) for the scoring target; the estimated t_res gives the target
    implied by the fitted ap model.
    """
    wt = df["dt_days"].to_numpy() * np.asarray(t_res) ** 2
    return float(-(df["true_tau"].to_numpy() * wt).sum() / wt.sum())


def true_leftover_ap(df, seed, n_draws=200, n_storms=20):
    """Synthetic data only: ap_mean - E[ap_mean | X] with the true conditional mean.

    In make_synthetic the Sun (f107, f107_81d) is drawn first, then the storms and the
    hourly ap given the Sun. Holding the Sun of `seed` fixed and redrawing storms and
    ap n_draws times gives the expected hourly ap at every hour; its mean over each
    interval is E[ap_mean | X] (with one Sun path, X pins down the date). df: rows from
    prepare(make_synthetic(seed=seed)[0]).
    """
    from src import synthetic as syn

    days = pd.date_range(syn.START, syn.END, freq="D")
    flux = syn.solar_flux(days, np.random.default_rng(seed))    # same first draw as make_synthetic
    expected = 0
    for m in range(n_draws):
        rng = np.random.default_rng([seed, 7, m])
        expected = expected + syn.hourly_ap(flux, syn.make_storms(flux, n_storms, rng), rng) / n_draws
    # Mean of the expected hourly ap over each interval, from its running integral
    hours = ((expected.index - syn.START).total_seconds() / 3600).to_numpy()
    grid = np.concatenate([hours, [hours[-1] + 1]])
    cum = np.concatenate([[0], np.cumsum(expected.to_numpy() / 24)])
    half = pd.to_timedelta(df["dt_days"] / 2, unit="D")
    t0 = ((df["t_mid"] - half - syn.START).dt.total_seconds() / 3600).to_numpy()
    t1 = ((df["t_mid"] + half - syn.START).dt.total_seconds() / 3600).to_numpy()
    mean_ap = (np.interp(t1, grid, cum) - np.interp(t0, grid, cum)) / df["dt_days"].to_numpy()
    return df[TREATMENT].to_numpy() - mean_ap


def compare(table, outcome="Y_residual", n_folds=5, n_boot=999, seed=0, all_t_models=True):
    """Naive, linear-control and DML estimates side by side.

    Returns a DataFrame (rows naive / linear / dml, plus dml_t_<name> for each ap
    model when all_t_models; columns coef, se_cr1, se_boot, ci_lo, ci_hi, n_clusters)
    with .attrs: sun_share = (naive - dml) / naive, the share of the naive effect that
    was really Sun UV and the other confounders; n_rows; model_y / model_t / selection;
    and planted_est (target implied by the fitted ap model) when the table has true_tau.
    The dml_t_<name> rows keep the selected residual model and force the ap model.
    """
    df = prepare(table, outcome)
    res = {"naive": naive(df, outcome, n_boot, seed),
           "linear": linear(df, outcome, n_boot, seed),
           "dml": dml(df, outcome, n_folds, n_boot, seed)}
    t_res = res["dml"].pop("t_res")
    res["dml"].pop("y_res")
    extra = {k: res["dml"].pop(k) for k in ["theta_econml", "model_y", "model_t", "selection"]}
    if all_t_models:
        for name in candidate_models():
            if name == extra["model_t"]:
                res[f"dml_t_{name}"] = dict(res["dml"])
                continue
            r = dml(df, outcome, n_folds, n_boot, seed, model_y=extra["model_y"], model_t=name)
            res[f"dml_t_{name}"] = {k: v for k, v in r.items()
                                    if k not in ("t_res", "y_res", "theta_econml", "model_y", "model_t", "selection")}
    out = pd.DataFrame(res).T
    out.attrs["sun_share"] = (out.loc["naive", "coef"] - out.loc["dml", "coef"]) / out.loc["naive", "coef"]
    out.attrs["n_rows"] = len(df)
    out.attrs.update(extra)
    if "true_tau" in df:
        out.attrs["planted_est"] = planted_theta(df, t_res)
    return out


def seed_study(n_seeds=50, n_objects=200, n_boot=499, all_t_models=True, **synthetic_kwargs):
    """Synthetic data only: run compare on n_seeds fake tables, selection inside the loop.

    Returns one row per seed: planted target (true leftover ap weights) and planted_est
    (fitted ap model weights), the chosen helper models, every method's coef and 95%
    CI, and whether each interval covers the planted target.
    """
    from src.synthetic import make_synthetic

    rows = []
    for seed in range(n_seeds):
        table, _ = make_synthetic(n_objects=n_objects, seed=seed, **synthetic_kwargs)
        res = compare(table, n_boot=n_boot, seed=seed, all_t_models=all_t_models)
        df = prepare(table)
        n_storms = synthetic_kwargs.get("n_storms", 20)
        row = {"seed": seed, "planted": planted_theta(df, true_leftover_ap(df, seed, n_storms=n_storms)),
               "planted_est": res.attrs["planted_est"], "sun_share": res.attrs["sun_share"],
               "model_y": res.attrs["model_y"], "model_t": res.attrs["model_t"]}
        for m in res.index:
            row[f"{m}_coef"] = res.loc[m, "coef"]
            row[f"{m}_lo"], row[f"{m}_hi"] = res.loc[m, "ci_lo"], res.loc[m, "ci_hi"]
            row[f"{m}_covers"] = res.loc[m, "ci_lo"] <= row["planted"] <= res.loc[m, "ci_hi"]
        rows.append(row)
    return pd.DataFrame(rows)
