// Phase 1 variant v4: lognormal observation model, both sides protected by the
// same P_min=5 floor:
//   log(Chla_obs + P_min) ~ normal(log(P_pred + P_min), sigma)
//
// This revisits the lognormal approach tried earlier (as part of npz_model.stan's
// history), which used P_min=2 and produced a distorted MAP (background pinned
// at its bound, ~7.76-15 depending on variant) -- hypothesized to be because a
// small P_min leaves the log-transform very steep near zero, so the few
// near-zero/negative fluorescence readings (dark-offset/blank-correction
// artifacts -- 8 rows as low as -0.59 in this dataset) get outsized leverage.
// Raising P_min to 5 softens that steepness on both sides. A separate
// pred-denominator normalized-residual variant was also tried in between and
// rejected: it let the optimizer game the objective by inflating `background`
// to shrink its own denominator, independent of fit quality -- a worse
// pathology than either lognormal attempt. This file returns to the lognormal
// family, not that normalized-residual family.
//
// Integration: Stan's adaptive ode_rk45_tol (not the fixed-step RK4 used in
// npz_model.stan) -- confirmed in this session's testing that the earlier
// near-degenerate MAP behavior was NOT caused by RK4 (the same distortion
// appeared under both integrators for the pred-denominator variant), so the
// adaptive solver is used here for its better optimize()-reliability, per
// this session's own findings on RK4's fragility.
//
// All 11 base parameters remain free (no fixed constants) -- this file is not
// a continuation of the v2/v3 parameter-fixing experiments, and separate from
// npz_model.stan's fixed-step RK4 configuration, to avoid confusing the two
// separate lines of experimentation (integration/sampling-speed vs.
// likelihood-form) that have been running in this session.

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
    // kLight is a light-limitation fraction and should not exceed 1; see
    // npz_model.stan's header for why this clamp is a correctness fix.
    return fmin(kLight, 1.0);
  }

  // state y = [N, P, Z]
  vector npz_ode(real t, vector y, real kprod, real kLight, real Nsat, real kmortP,
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
  real<lower=0> P_min;     // floor added to both obs and pred before taking log (5.0)
}

transformed data {
  // Parameters held fixed at NPZIVModel.base_params defaults for this phase --
  // not part of the original "zooplankton first go" tuned parameter set.
  real kmortZ = 0.2;
  real Z_efficiency = 0.3;
  real mortP_to_N_efficiency = 0.5;
  real alpha = 0.15;
  real Ivlev = 1.0;

  // Same +P_min floor used on the predicted side, applied here so both sides
  // of the likelihood are on a consistent, strictly-positive scale (handles
  // the dataset's near-zero/negative fluorescence readings).
  vector[N_obs] log_Chla_obs = log(Chla_obs + P_min);
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
      // Bounded rel_tol/abs_tol/max_num_steps: see npz_model_v2/v3.stan for
      // why this cap is kept as a safety net even with kLight clamped.
      array[1] vector[3] ode_sol = ode_rk45_tol(npz_ode, y0, 0.0, {age_d[n]},
                                                 1e-4, 1e-4, 500,
                                                 kprod, kLight, Nsat, kmortP,
                                                 Z_max_ration, Ivlev, Z_efficiency,
                                                 kmortZ, alpha, mortP_to_N_efficiency,
                                                 N_supplement);
      state = ode_sol[1];
    } else {
      state = y0;
    }
    P_pred[n] = state[2] + background;
  }
}

model {
  sigma ~ normal(0, 1); // half-normal, sigma>0 by declared bound; on the log scale
  // remaining parameters: implicit uniform prior over their declared bounds,
  // mirroring the original scipy bounds used with method='powell'.

  for (n in 1:N_obs) {
    log_Chla_obs[n] ~ normal(log(P_pred[n] + P_min), sigma);
  }
}

generated quantities {
  // Directly comparable to the loss() values printed in the scipy version --
  // kept as a diagnostic only; it does not drive the likelihood above.
  real loss = 0;
  for (n in 1:N_obs) {
    real resid = (Chla_obs[n] - P_pred[n]) / (Chla_obs[n] + P_min);
    loss += resid^2;
  }
  loss /= N_obs;
}
