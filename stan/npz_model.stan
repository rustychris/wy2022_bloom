// Phase 1: NPZ (nutrient-phytoplankton-zooplankton) model, no virus terms.
// Reimplements NPZIVModel.npziv_diff (bloom_lagrangian_zoop_etal_opt_20260729.py) with
// I=V=0 fixed throughout, matching the "zooplankton first go" scipy fit
// (params_to_tune at that script's line ~749).
//
// kLight cannot be precomputed as data: it depends on the Isat parameter via the
// log-linear interpolation in bloom_common.calc_kLight, so it is recomputed here
// from kRadA/IsatA/kRadB/IsatB/conc/age_d on every evaluation.
//
// Integration: fixed-step RK4, not Stan's adaptive ode_rk45/ode_rk45_tol.
// Rationale (see conversation history / commit notes): the declared parameter
// bounds keep the Jacobian's eigenvalues to ~O(10)/day, i.e. not stiff, so a
// fixed low-order scheme is appropriate and far cheaper than an adaptive
// integrator -- adaptive solvers were seeing occasional HMC warmup proposals
// (large Isat extrapolating kLight well above 1 via the log-linear formula
// below) drive transient explosive growth, forcing huge/unpredictable internal
// step counts and stalling NUTS. Clamping kLight to <=1 (its intended meaning:
// a light-limitation *fraction*) removes that explosive pathway at the source,
// making a fixed-step, fixed-cost-per-row integrator safe. Per-row step count
// is precomputed once from data (age_d, dt_target), not re-evaluated per
// leapfrog, so cost per row is now deterministic and uniform.

functions {
  real klight_interp(real Isat, real kRadA, real IsatA, real kRadB, real IsatB,
                      real conc, real age_d, real thresh) {
    real ageconc = conc * age_d;
    real kLightA = kRadA / fmax(ageconc, thresh);
    real kLightB = kRadB / fmax(ageconc, thresh);
    real log_kLightA = log(fmin(fmax(kLightA, 1e-5), 1.0));
    real log_kLightB = log(fmin(fmax(kLightB, 1e-5), 1.0));
    real slope = (log_kLightB - log_kLightA) / (IsatB - IsatA);
    real log_kLight0 = log_kLightA - slope * IsatA;
    real kLight = exp(log_kLight0 + slope * Isat);
    // kLight is a light-limitation fraction and should not exceed 1; the
    // log-linear extrapolation above is unclipped for Isat outside [IsatA,
    // IsatB], which is what let occasional HMC proposals drive it far above 1
    // and destabilize the ODE. Clamping here is a correctness fix as much as
    // a performance one.
    return fmin(kLight, 1.0);
  }

  // state y = [N, P, Z]. Autonomous (no explicit t-dependence), so a plain
  // fixed-step RK4 needs only the state and shared parameters, not time.
  vector npz_rhs(vector y, real kprod, real kLight, real Nsat, real kmortP,
                 real Z_max_ration, real Ivlev, real Z_efficiency, real kmortZ,
                 real alpha, real mortP_to_N_efficiency, real N_supplement) {
    real N = fmax(y[1], 0);
    real P = fmax(y[2], 0);
    real Z = fmax(y[3], 0);

    real limDIN = N / (N + Nsat);
    real dgrossP = kprod * kLight * limDIN * P;
    real dmortP = kmortP * P;
    real dgrazeP = Z * Z_max_ration * (1 - exp(-Ivlev * P));
    real dnetP = dgrossP - dmortP - dgrazeP;
    real dZ = Z_efficiency * dgrazeP - kmortZ * Z;
    real dN = -alpha * dgrossP
              + mortP_to_N_efficiency * alpha * dmortP
              + alpha * (1 - Z_efficiency) * dgrazeP
              + N_supplement;
    return [dN, dnetP, dZ]';
  }

  vector npz_rk4(vector y0, real h, int n_steps, real kprod, real kLight, real Nsat,
                 real kmortP, real Z_max_ration, real Ivlev, real Z_efficiency,
                 real kmortZ, real alpha, real mortP_to_N_efficiency, real N_supplement) {
    vector[3] y = y0;
    for (s in 1:n_steps) {
      vector[3] k1 = npz_rhs(y, kprod, kLight, Nsat, kmortP, Z_max_ration, Ivlev,
                              Z_efficiency, kmortZ, alpha, mortP_to_N_efficiency, N_supplement);
      vector[3] k2 = npz_rhs(y + 0.5 * h * k1, kprod, kLight, Nsat, kmortP, Z_max_ration,
                              Ivlev, Z_efficiency, kmortZ, alpha, mortP_to_N_efficiency, N_supplement);
      vector[3] k3 = npz_rhs(y + 0.5 * h * k2, kprod, kLight, Nsat, kmortP, Z_max_ration,
                              Ivlev, Z_efficiency, kmortZ, alpha, mortP_to_N_efficiency, N_supplement);
      vector[3] k4 = npz_rhs(y + h * k3, kprod, kLight, Nsat, kmortP, Z_max_ration,
                              Ivlev, Z_efficiency, kmortZ, alpha, mortP_to_N_efficiency, N_supplement);
      y += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    return y;
  }
}

data {
  int<lower=0> N_obs;
  vector[N_obs] conc;      // tracer concentration at release, scaled by c0 for IC
  vector[N_obs] age_d;     // tracer age in days at time of sample
  vector[N_obs] kRadA;
  vector[N_obs] IsatA;
  vector[N_obs] kRadB;
  vector[N_obs] IsatB;
  vector[N_obs] Chla_obs;  // observed Chl-a, ug/l

  real<lower=0> thresh;    // ageconc floor before kLight is considered undefined (1e-5)
  real<lower=0> P_min;     // normalization floor in the residual (2.0, matches loss())

  // Fixed-step RK4 target step size, days. Declared as data (not hardcoded) so
  // it can be tuned from the driver script without recompiling the model.
  real<lower=0> dt_target;
}

transformed data {
  // Parameters held fixed at NPZIVModel.base_params defaults for this phase --
  // not part of the original "zooplankton first go" tuned parameter set.
  real kmortZ = 0.2;
  real Z_efficiency = 0.3;
  real mortP_to_N_efficiency = 0.5;
  real alpha = 0.15;
  real Ivlev = 1.0;

  // Precomputed once from data: number of fixed RK4 steps per row so the
  // physical step size stays close to dt_target regardless of age_d.
  array[N_obs] int<lower=1> n_steps;
  for (n in 1:N_obs) {
    n_steps[n] = age_d[n] > 0 ? max(1, to_int(ceil(age_d[n] / dt_target))) : 1;
  }
}

parameters {
  real<lower=0.5, upper=5> c0;
  real<lower=0.5, upper=2.0> kprod;
  real<lower=0.05, upper=2> N_supplement;
  real<lower=5, upper=40> N0;
  real<lower=0.00001, upper=0.001> Z0;
  real<lower=0.1, upper=4.0> Z_max_ration;
  real<lower=0.01, upper=0.5> kmortP;
  real<lower=1.0, upper=80> Isat;
  real<lower=0.5, upper=15.0> background;
  real<lower=0.1, upper=5.0> Nsat;
  real<lower=0> sigma;
}

transformed parameters {
  vector[N_obs] P_pred;

  for (n in 1:N_obs) {
    real kLight = klight_interp(Isat, kRadA[n], IsatA[n], kRadB[n], IsatB[n],
                                 conc[n], age_d[n], thresh);
    vector[3] y0 = [N0, c0 * conc[n], Z0]';
    vector[3] state;

    if (age_d[n] > 0) {
      real h = age_d[n] / n_steps[n];
      state = npz_rk4(y0, h, n_steps[n], kprod, kLight, Nsat, kmortP, Z_max_ration,
                       Ivlev, Z_efficiency, kmortZ, alpha, mortP_to_N_efficiency, N_supplement);
    } else {
      state = y0;
    }
    P_pred[n] = state[2] + background;
  }
}

model {
  // NOTE: two alternative likelihoods were tried and rejected here -- see
  // stan/npz_model_v4.stan's header for the current best candidate and why.
  //  - (obs-pred)/(pred+P_min) ~ normal(0,sigma): fixes the backwards
  //    heteroscedasticity below, but lets the optimizer game the objective by
  //    inflating `background` to shrink its own denominator (pred appears in
  //    the normalizer, so a bigger prediction directly shrinks the residual
  //    regardless of fit quality) -- pinned background to its bound (15) with
  //    a much worse fit (loss~2-760) under BOTH RK4 and the adaptive solver.
  //  - A log-scale (lognormal / Student-t) likelihood, motivated by
  //    chlorophyll fluorescence data being positive/right-skewed/roughly
  //    constant-CV -- also pinned background to its bound with P_min=2 (see
  //    v4 for the retry with P_min=5, which is more promising).
  // This form's known flaw (noise scale driven by the *observed* value,
  // rather than latent/predicted -- backwards for a measurement-error model)
  // is real, but neither alternative tried so far is a practical improvement,
  // so it's kept for now as the working baseline.
  sigma ~ normal(0, 1); // half-normal, sigma>0 by declared bound
  // remaining parameters: implicit uniform prior over their declared bounds,
  // mirroring the original scipy bounds used with method='powell'.

  for (n in 1:N_obs) {
    real resid = (Chla_obs[n] - P_pred[n]) / (Chla_obs[n] + P_min);
    resid ~ normal(0, sigma);
  }
}

generated quantities {
  // Directly comparable to the loss() values printed in the scipy version.
  real loss = 0;
  for (n in 1:N_obs) {
    real resid = (Chla_obs[n] - P_pred[n]) / (Chla_obs[n] + P_min);
    loss += resid^2;
  }
  loss /= N_obs;
}
