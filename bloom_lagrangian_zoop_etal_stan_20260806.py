"""
Stan reimplementation of the parameter-tuning step in
bloom_lagrangian_zoop_etal_opt_20260729.py.

Phase 1: NPZ-only (no virus terms), matching that script's "zooplankton first go"
scipy/Powell fit. Reads a pre-extracted observation/tracer CSV (produced by the
opt script's `model.df_inputs.to_csv(...)` call) rather than re-deriving it from
DFM/DWAQ output or mooring data -- this script has no dependency on stompy,
tracer_group.py, or bloom_common.py.

Usage:
    python bloom_lagrangian_zoop_etal_stan_20260806.py [--csv PATH] [--mode optimize|sample|both]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cmdstanpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STAN_MODEL_FILE = os.path.join(THIS_DIR, "stan", "npz_model.stan")
DEFAULT_CSV = os.path.join(THIS_DIR, "stan", "inputs-v17-conc1_6.9mpd_alameda.csv")

# Matches calc_kLight's default `thresh` and LagrangianModel.min_ageconc.
KLIGHT_THRESH = 1e-5
# Matches loss()'s P_min. Some model variants (e.g. npz_model_v4.stan) use a
# different value for their actual likelihood's floor -- pass --p-min to
# override per run rather than changing this default.
P_MIN = 2.0
# Fixed-step RK4 target step size (days). See npz_model.stan's header comment
# for why a fixed-step scheme is appropriate here and how this was chosen.
DT_TARGET = 0.1

# Starting point for optimize(): the scipy/Powell optimum reported in
# bloom_lagrangian_zoop_etal_opt_20260729.py (loss ~0.3792, "zooplankton first go").
SCIPY_REFERENCE_FIT = dict(
    c0=4.506984204870644,
    kprod=1.1243508317602493,
    N_supplement=0.16631002953092275,
    N0=9.931855274572516,
    Z0=0.0002972396053085654,
    Z_max_ration=2.0980713339308092,
    kmortP=0.053263070972951444,
    Isat=3.697702281261708,
    background=1.6387620761482578,
    Nsat=0.46703945899618593,
)
SCIPY_REFERENCE_LOSS = 0.3792


def load_stan_data(csv_path, dt_target=DT_TARGET, p_min=P_MIN):
    """Load a df_inputs-style CSV and build the data dict for npz_model.stan."""
    df = pd.read_csv(csv_path, parse_dates=["time", "t"])

    ageconc = df["conc"] * df["age_d"]
    valid = ageconc >= KLIGHT_THRESH
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(df)} rows with conc*age_d < {KLIGHT_THRESH:g}"
              " (undefined kLight)")
    df = df.loc[valid].reset_index(drop=True)

    data = dict(
        N_obs=len(df),
        conc=df["conc"].values,
        age_d=df["age_d"].values,
        kRadA=df["kRadA"].values,
        IsatA=df["IsatA"].values,
        kRadB=df["kRadB"].values,
        IsatB=df["IsatB"].values,
        Chla_obs=df["Chla_microgL"].values,
        thresh=KLIGHT_THRESH,
        P_min=p_min,
        dt_target=dt_target,
    )
    return df, data


def fitted_param_names(model):
    """Names actually declared in this model's `parameters` block -- lets one
    driver script work across model variants with different tuned/fixed
    parameter sets (e.g. npz_model.stan vs. npz_model_v2.stan)."""
    return list(model.src_info().get("parameters", {}).keys())


def parse_stan_bounds(stan_file):
    """Crude parse of `real<lower=..., upper=...> name;` declarations in the
    `parameters` block, so an init value from a different model variant can be
    clipped into range rather than erroring out (variants may both narrow and
    shift a parameter's bounds relative to where the reference fit landed)."""
    text = open(stan_file).read()
    block_match = re.search(r"\bparameters\s*\{(.*?)\n\}", text, re.S)
    block = block_match.group(1) if block_match else ""
    bounds = {}
    for m in re.finditer(r"real(?:<([^>]*)>)?\s+(\w+)\s*;", block):
        spec, name = m.group(1), m.group(2)
        lower, upper = -np.inf, np.inf
        if spec:
            lo = re.search(r"lower\s*=\s*([^,]+)", spec)
            hi = re.search(r"upper\s*=\s*([^,]+)", spec)
            if lo:
                lower = float(lo.group(1))
            if hi:
                upper = float(hi.group(1))
        bounds[name] = (lower, upper)
    return bounds


def reference_inits(model, stan_file):
    """SCIPY_REFERENCE_FIT restricted to this model's actual parameters and
    clipped into its declared bounds -- a variant (e.g. npz_model_v2.stan) may
    have narrowed or shifted bounds so the raw reference value no longer fits."""
    names = fitted_param_names(model)
    bounds = parse_stan_bounds(stan_file)
    inits = {}
    for k, v in SCIPY_REFERENCE_FIT.items():
        if k not in names:
            continue
        lower, upper = bounds.get(k, (-np.inf, np.inf))
        if np.isfinite(lower) or np.isfinite(upper):
            margin = 1e-6 * (upper - lower) if np.isfinite(lower) and np.isfinite(upper) else 0
            v = min(max(v, lower + margin), upper - margin)
        inits[k] = v
    if "sigma" in names:
        inits["sigma"] = 0.3
    return inits


def random_inits(model, stan_file, rng):
    """Uniform draw within each parameter's declared bounds (for the rare
    one-sided/unbounded case, e.g. sigma, fall back to a fixed reasonable
    default rather than sampling an unbounded range)."""
    names = fitted_param_names(model)
    bounds = parse_stan_bounds(stan_file)
    inits = {}
    for name in names:
        lower, upper = bounds.get(name, (-np.inf, np.inf))
        if np.isfinite(lower) and np.isfinite(upper):
            inits[name] = rng.uniform(lower, upper)
        elif name == "sigma":
            inits[name] = rng.uniform(0.1, 2.0)
        else:
            inits[name] = rng.uniform(max(lower, -10), min(upper, 10))
    return inits


# Looser than cmdstanpy's defaults (tol_obj=1e-12 etc.) -- the fixed-step RK4's
# small, non-adaptive truncation error makes the objective slightly rougher near
# an optimum than the old adaptive-tolerance solver did, which was tripping
# L-BFGS's line search ("no more progress can be made") well before that level
# of precision matters relative to sigma's noise floor.
LBFGS_TOL_KWARGS = dict(tol_obj=1e-8, tol_rel_obj=1e4, tol_grad=1e-6,
                        tol_rel_grad=1e6, tol_param=1e-6)


def run_optimize(model, data, stan_file):
    inits = reference_inits(model, stan_file)
    mle = model.optimize(data=data, inits=inits, algorithm="lbfgs", **LBFGS_TOL_KWARGS)
    fitted = {k: mle.stan_variable(k) for k in fitted_param_names(model)}
    loss = float(mle.stan_variable("loss"))
    lp = float(mle.optimized_params_dict["lp__"])
    print(f"cmdstanpy optimize(): lp__={lp:.3f}  loss={loss:.4f}  (scipy/Powell reference: "
          f"{SCIPY_REFERENCE_LOSS:.4f})")
    for k, v in fitted.items():
        ref = f"   (scipy: {SCIPY_REFERENCE_FIT[k]:.6g})" if k in SCIPY_REFERENCE_FIT else ""
        print(f"  {k:16s} = {v:.6g}{ref}")
    return mle, fitted, loss


def run_multistart_optimize(model, data, stan_file, n_starts=10, seed=0):
    """Run optimize() from n_starts random initial points (plus the usual
    reference-fit init) and report the result with the highest actual
    log-density (lp__) -- a cheap way to probe for a better local optimum
    than a single L-BFGS run finds, given this loss surface is known to have
    multiple comparably-good basins (see bloom_lagrangian_zoop_etal_opt_20260729.py's
    own exploratory notes).

    IMPORTANT: ranks by lp__ (Stan's actual fitted objective), not by the
    loss() diagnostic. The two do not always agree -- e.g. on npz_model_v5.stan
    a candidate with loss()=0.589 was verified (via model.log_prob()) to have
    a dramatically higher lp__ than another candidate with the "better"
    loss()=0.314. loss() exists only for rough cross-reference to the
    original scipy script's metric; it is not the objective being fit and
    should not be used to rank candidates."""
    rng = np.random.default_rng(seed)
    candidates = [("reference", reference_inits(model, stan_file))]
    candidates += [(f"random_{i}", random_inits(model, stan_file, rng)) for i in range(n_starts)]

    results = []
    for label, inits in candidates:
        try:
            mle = model.optimize(data=data, inits=inits, algorithm="lbfgs", **LBFGS_TOL_KWARGS)
            loss = float(mle.stan_variable("loss"))
            lp = float(mle.optimized_params_dict["lp__"])
        except RuntimeError as exc:
            print(f"  [{label}] failed: {exc}", flush=True)
            continue
        print(f"  [{label}] lp__={lp:.3f}  loss={loss:.4f}", flush=True)
        results.append((lp, loss, label, mle))

    results.sort(key=lambda r: r[0], reverse=True)  # highest lp__ (best fit) first
    best_lp, best_loss, best_label, best_mle = results[0]
    fitted = {k: best_mle.stan_variable(k) for k in fitted_param_names(model)}
    print(f"\nBest of {len(results)} starts: [{best_label}] lp__={best_lp:.3f}  "
          f"loss={best_loss:.4f}  (scipy/Powell reference: {SCIPY_REFERENCE_LOSS:.4f})")
    for k, v in fitted.items():
        ref = f"   (scipy: {SCIPY_REFERENCE_FIT[k]:.6g})" if k in SCIPY_REFERENCE_FIT else ""
        print(f"  {k:16s} = {v:.6g}{ref}")
    return best_mle, fitted, best_loss


def run_sample(model, data, inits, chains=4, iter_warmup=500, iter_sampling=500, max_treedepth=10):
    # Each leapfrog step re-solves the ODE for every observation, so a NUTS
    # trajectory that hits max_treedepth (up to 2**max_treedepth - 1 leapfrog
    # steps) can be very slow. Capping max_treedepth bounds worst-case cost per
    # iteration; see stan/npz_model.stan's bounded ode_rk45_tol for the same
    # reasoning applied to a single ODE solve.
    fit = model.sample(
        data=data,
        inits=inits,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        max_treedepth=max_treedepth,
        show_progress=True,
    )
    print(fit.summary().loc[fitted_param_names(model) + ["loss"]])
    return fit


def fig_time_series(df, P_pred, site=None):
    """Reproduces fig_time_series() from the opt script, but driven entirely by
    the CSV + this model's predictions -- no tracer_group/grid access needed."""
    if site is not None:
        mask = (df["site"] == site).values
    else:
        mask = np.ones(len(df), dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df["time"][mask], df["Chla_microgL"][mask], "o-", label=f"{site or 'all'} mooring")
    ax.plot(df["time"][mask], np.asarray(P_pred)[mask], "o-", label="Stan NPZ model")
    ax.set_ylabel("Chl-a (ug/l)")
    ax.legend(loc="upper left")
    ax.axis(ymin=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def save_or_show(fig, out_dir, name):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=110)
        print(f"wrote {path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--stan-file", default=STAN_MODEL_FILE,
                         help="Path to a .stan model file (default: stan/npz_model.stan)")
    parser.add_argument("--mode", choices=["optimize", "sample", "both"], default="optimize")
    parser.add_argument("--out-dir", default=None,
                         help="If set, save diagnostic plots here instead of calling plt.show()")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=500)
    parser.add_argument("--iter-sampling", type=int, default=500)
    parser.add_argument("--max-treedepth", type=int, default=10)
    parser.add_argument("--multistart", type=int, default=0,
                         help="If >0, run optimize() from this many random starts (plus the "
                              "reference-fit start) and keep the best (lowest-loss) result.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt-target", type=float, default=DT_TARGET,
                         help="Fixed-step RK4 target step size, days")
    parser.add_argument("--p-min", type=float, default=P_MIN,
                         help="Floor added before normalizing/logging observed and predicted "
                              "Chl-a in the likelihood")
    args = parser.parse_args()

    df, data = load_stan_data(args.csv, dt_target=args.dt_target, p_min=args.p_min)
    model = cmdstanpy.CmdStanModel(stan_file=args.stan_file)

    if args.mode in ("optimize", "both"):
        if args.multistart > 0:
            mle, fitted, loss = run_multistart_optimize(model, data, args.stan_file,
                                                         n_starts=args.multistart, seed=args.seed)
        else:
            mle, fitted, loss = run_optimize(model, data, args.stan_file)
        P_pred = mle.stan_variable("P_pred")
        for site in sorted(df["site"].unique()):
            fig = fig_time_series(df, P_pred, site=site)
            save_or_show(fig, args.out_dir, f"optimize_{site}.png")

    if args.mode in ("sample", "both"):
        inits = reference_inits(model, args.stan_file)
        fit = run_sample(model, data, inits=inits, chains=args.chains,
                          iter_warmup=args.iter_warmup, iter_sampling=args.iter_sampling,
                          max_treedepth=args.max_treedepth)
        P_pred = fit.stan_variable("P_pred").mean(axis=0)
        for site in sorted(df["site"].unique()):
            fig = fig_time_series(df, P_pred, site=site)
            save_or_show(fig, args.out_dir, f"sample_{site}.png")


if __name__ == "__main__":
    main()
