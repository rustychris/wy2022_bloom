"""
Identifiability audit for the Phase 1 NPZ model (matches stan/npz_model_v4.stan:
lognormal likelihood, log(Chla_obs+P_min) ~ normal(log(P_pred+P_min), sigma),
P_min=5, kLight clamped to <=1) -- this revision reparametrizes away the
kprod/Isat confounding and the Nsat weak-identification found in the first
audit pass:
  - kprod fixed at a literature value (1.7), not fit. The first audit's
    sloppy-direction analysis found kprod/Isat as the single sloppiest
    (least identified) combination -- they enter the model only through
    kprod*kLight(Isat), a classic growth-rate/light-saturation confounding.
    Fixing kprod tests whether Isat is then pinned at its lower bound (which
    would indicate the data wants an even lower effective growth rate than
    kprod=1.7 alone allows, i.e. that this particular literature value may be
    too high given the rest of the model) or settles to an interior value.
  - Nsat fixed at a literature value (1.8), not fit -- the first audit found
    it weakly identified on its own (data barely constrains it once N0 is
    comfortably above it).
  - Z0's lower bound loosened (1e-5 -> 1e-8). The first audit found Z0 was
    simultaneously pinned at its old lower bound (1e-5) AND the single
    stiffest (best-identified) direction in the whole spectrum -- i.e. not an
    identifiability problem but a bounds problem: the data has a genuine,
    sharp preference for Z0 smaller than the bound allowed.

Reimplements the forward dynamics and likelihood directly in Python/numpy,
with the per-row ODE integration fully vectorized (fixed-step RK4, same
step count for every row), so a large multistart search and a local
Fisher-information/sloppy-direction analysis can run in seconds rather than
requiring repeated Stan compiles/subprocess launches. This is not a
replacement for the Stan model -- it's a fast proxy over the identical
objective, used purely to characterize identifiability before deciding what
to reparametrize or fix in the Stan model. Kept in stan/ alongside the .stan
model files it supports, even though it's plain Python, not Stan.

Usage:
    python stan/identifiability_audit.py [--n-starts 40] [--seed 0]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bloom_lagrangian_zoop_etal_stan_20260806 import DEFAULT_CSV, KLIGHT_THRESH

PARAM_NAMES = ["c0", "N_supplement", "N0", "Z0", "Z_max_ration",
               "kmortP", "Isat", "background", "sigma"]
BOUNDS = dict(
    c0=(0.5, 5), N_supplement=(0.05, 2), N0=(5, 40),
    Z0=(1e-8, 0.001),  # lower bound loosened from 1e-5 per the first audit's findings
    Z_max_ration=(0.1, 4.0), kmortP=(0.01, 0.5),
    Isat=(1.0, 80), background=(0.5, 15.0), sigma=(1e-3, 5.0),
)
# kprod and Nsat fixed at literature values (not fit) -- see module docstring.
FIXED = dict(kmortZ=0.2, Z_efficiency=0.3, mortP_to_N_efficiency=0.5, alpha=0.15,
             Ivlev=1.0, kprod=1.7, Nsat=1.8)
P_MIN = 5.0
N_STEPS = 100  # fixed RK4 steps per row, same for every row regardless of age_d


def klight(Isat, kRadA, IsatA, kRadB, IsatB, conc, age_d, thresh=KLIGHT_THRESH):
    ageconc = conc * age_d
    kLightA = kRadA / np.maximum(ageconc, thresh)
    kLightB = kRadB / np.maximum(ageconc, thresh)
    log_a = np.log(np.clip(kLightA, 1e-5, 1.0))
    log_b = np.log(np.clip(kLightB, 1e-5, 1.0))
    slope = (log_b - log_a) / (IsatB - IsatA)
    log0 = log_a - slope * IsatA
    return np.minimum(np.exp(log0 + slope * Isat), 1.0)


def npz_rhs(state, kL, kmortP, Zmax):
    # state: (N_obs, 3) = [N, P, Z]; kL, per-row: (N_obs,)
    N = np.maximum(state[:, 0], 0)
    P = np.maximum(state[:, 1], 0)
    Z = np.maximum(state[:, 2], 0)

    lim_din = N / (N + FIXED["Nsat"])
    dgross_p = FIXED["kprod"] * kL * lim_din * P
    dmort_p = kmortP * P
    dgraze_p = Z * Zmax * (1 - np.exp(-FIXED["Ivlev"] * P))
    dnet_p = dgross_p - dmort_p - dgraze_p
    dz = FIXED["Z_efficiency"] * dgraze_p - FIXED["kmortZ"] * Z
    dn = (-FIXED["alpha"] * dgross_p
          + FIXED["mortP_to_N_efficiency"] * FIXED["alpha"] * dmort_p
          + FIXED["alpha"] * (1 - FIXED["Z_efficiency"]) * dgraze_p)
    return np.stack([dn, dnet_p, dz], axis=1)


def forward(params, df):
    """Vectorized fixed-step RK4 across all rows at once. Returns P_pred (N_obs,)."""
    p = dict(zip(PARAM_NAMES, params))
    conc = df["conc"].values
    age_d = df["age_d"].values
    kL = klight(p["Isat"], df["kRadA"].values, df["IsatA"].values,
                df["kRadB"].values, df["IsatB"].values, conc, age_d)

    y0 = np.stack([np.full(len(df), p["N0"]), p["c0"] * conc,
                    np.full(len(df), p["Z0"])], axis=1)
    h = np.where(age_d > 0, age_d / N_STEPS, 0.0)[:, None]  # (N_obs,1)

    def rhs(y):
        base = npz_rhs(y, kL, p["kmortP"], p["Z_max_ration"])
        base[:, 0] += p["N_supplement"]  # N_supplement is a constant source term
        return base

    y = y0.copy()
    for _ in range(N_STEPS):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    P_pred = np.where(age_d > 0, y[:, 1], y0[:, 1]) + p["background"]
    return P_pred


def neg_log_post(params, df, log_obs):
    p = dict(zip(PARAM_NAMES, params))
    P_pred = forward(params, df)
    resid = (log_obs - np.log(P_pred + P_MIN)) / p["sigma"]
    nll = 0.5 * np.sum(resid ** 2) + len(df) * np.log(p["sigma"])
    nll += 0.5 * p["sigma"] ** 2  # half-normal(0,1) prior on sigma, matches Stan
    return nll


def loss_metric(params, df):
    """Diagnostic only -- NOT the fitted objective. Uses P_MIN (5.0, matching
    this audit's likelihood) rather than the original scipy P_min=2.0, so it's
    directly comparable to the loss= values Stan's v4 runs report, not to the
    original scipy script's loss()."""
    P_pred = forward(params, df)
    resid = (df["Chla_microgL"].values - P_pred) / (df["Chla_microgL"].values + P_MIN)
    return float(np.mean(resid ** 2))


def load_data(csv_path=DEFAULT_CSV):
    df = pd.read_csv(csv_path, parse_dates=["time", "t"])
    ageconc = df["conc"] * df["age_d"]
    df = df.loc[ageconc >= KLIGHT_THRESH].reset_index(drop=True)
    log_obs = np.log(df["Chla_microgL"].values + P_MIN)
    return df, log_obs


def random_start(rng):
    return np.array([rng.uniform(*BOUNDS[name]) for name in PARAM_NAMES])


def reference_start():
    ref = dict(c0=4.51, N_supplement=0.166, N0=9.93, Z0=0.000297,
               Z_max_ration=2.098, kmortP=0.0533, Isat=3.70, background=1.64,
               sigma=0.6)
    return np.array([ref[name] for name in PARAM_NAMES])


def run_multistart(df, log_obs, n_starts, seed):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    bounds = [BOUNDS[name] for name in PARAM_NAMES]

    starts = [("reference", reference_start())]
    starts += [(f"random_{i}", random_start(rng)) for i in range(n_starts)]

    results = []
    for label, x0 in starts:
        res = minimize(neg_log_post, x0, args=(df, log_obs), method="L-BFGS-B",
                        bounds=bounds)
        results.append(dict(label=label, success=res.success, nll=res.fun,
                             x=res.x, loss=loss_metric(res.x, df)))
    return results


def summarize(results, tol_frac=0.02):
    ok = [r for r in results if r["success"] and np.isfinite(r["nll"])]
    print(f"{len(ok)}/{len(results)} starts converged")
    if not ok:
        return

    ok.sort(key=lambda r: r["nll"])
    best = ok[0]
    print(f"\nBest: nll={best['nll']:.3f}  loss()={best['loss']:.4f}")
    for name, v in zip(PARAM_NAMES, best["x"]):
        print(f"  {name:16s} = {v:.6g}")
    print(f"  (fixed) kprod    = {FIXED['kprod']:.6g}")
    print(f"  (fixed) Nsat     = {FIXED['Nsat']:.6g}")

    # "good enough" set: within 2 nats of the best negative-log-posterior
    # (roughly: not decisively rejected relative to the best fit)
    good = [r for r in ok if r["nll"] - best["nll"] < 2.0]
    print(f"\n{len(good)}/{len(ok)} converged starts are within 2 nats of the best "
          "(a 'comparably good' set for the checks below)")

    print("\nParameter spread across the comparably-good set "
          "(wide spread = weakly identified by this data):")
    xs = np.array([r["x"] for r in good])
    lo = np.array([BOUNDS[n][0] for n in PARAM_NAMES])
    hi = np.array([BOUNDS[n][1] for n in PARAM_NAMES])
    frac_range = (xs.max(axis=0) - xs.min(axis=0)) / (hi - lo)
    for name, f, lo_i, hi_i in zip(PARAM_NAMES, frac_range, xs.min(axis=0), xs.max(axis=0)):
        print(f"  {name:16s} spans {f*100:5.1f}% of its bound range  "
              f"[{lo_i:.4g}, {hi_i:.4g}]")

    print("\nBound-hit frequency across the comparably-good set "
          "(near a bound = pinned, a red flag either way):")
    for j, name in enumerate(PARAM_NAMES):
        lo_j, hi_j = BOUNDS[name]
        tol = tol_frac * (hi_j - lo_j)
        at_lower = np.mean(xs[:, j] - lo_j < tol)
        at_upper = np.mean(hi_j - xs[:, j] < tol)
        if at_lower > 0 or at_upper > 0:
            print(f"  {name:16s} at lower bound {at_lower*100:5.1f}% of the time, "
                  f"at upper bound {at_upper*100:5.1f}% of the time")

    return best["x"]


def sloppy_direction_analysis(best_x, df, log_obs, rel_step=1e-4):
    """Gauss-Newton Hessian of the log-scale mean function at the MAP, in
    fractional-of-bound-range units (so parameters with very different
    natural scales, e.g. Z0~1e-5 vs N0~10s, are comparable). Small
    eigenvalues => 'sloppy' directions/combinations the data barely
    constrains; large eigenvalues => well-identified combinations."""
    n = len(PARAM_NAMES)
    ranges = np.array([BOUNDS[name][1] - BOUNDS[name][0] for name in PARAM_NAMES])

    def g(params):
        return np.log(forward(params, df) + P_MIN)

    g0 = g(best_x)
    J = np.zeros((len(df), n))
    for j in range(n):
        step = rel_step * ranges[j]
        x_plus = best_x.copy()
        x_plus[j] += step
        J[:, j] = (g(x_plus) - g0) / step * ranges[j]  # scale back to per-unit-range

    sigma = best_x[PARAM_NAMES.index("sigma")]
    H = (J.T @ J) / sigma ** 2

    eigvals, eigvecs = np.linalg.eigh(H)
    order = np.argsort(eigvals)  # ascending: sloppiest first
    print("\nSloppy-direction spectrum (Gauss-Newton Hessian eigenvalues, "
          "fractional-of-range units; ascending = sloppiest first):")
    for rank, idx in enumerate(order):
        vec = eigvecs[:, idx]
        top = np.argsort(-np.abs(vec))[:4]
        loadings = ", ".join(f"{PARAM_NAMES[k]}={vec[k]:+.2f}" for k in top)
        print(f"  eigenvalue[{rank:2d}] = {eigvals[idx]:.3e}   dominant loadings: {loadings}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--n-starts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, log_obs = load_data(args.csv)
    print(f"Loaded {len(df)} rows")

    results = run_multistart(df, log_obs, args.n_starts, args.seed)
    best_x = summarize(results)
    if best_x is not None:
        sloppy_direction_analysis(best_x, df, log_obs)


if __name__ == "__main__":
    main()
