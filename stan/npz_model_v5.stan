// Phase 1 variant v5: same lognormal likelihood as npz_model_v4.stan
// (log(Chla_obs+P_min) ~ normal(log(P_pred+P_min), sigma), P_min=5, adaptive
// ode_rk45_tol, kLight clamped to <=1), reparametrized per the
// stan/identifiability_audit.py findings on v4's fitted parameters:
//
//   kprod: fixed at a literature value, 1.7 (was fit in [0.5, 2.0]). The
//     audit's sloppy-direction analysis found kprod/Isat as the single
//     sloppiest (least identified) combination -- they enter the model only
//     through kprod*kLight(Isat), a classic growth-rate/light-saturation
//     confounding. Verified: re-running the audit with kprod fixed does NOT
//     push Isat to its lower bound -- it settles at Isat=3.445, close to the
//     original scipy reference (3.70), confirming this isn't just displacing
//     the confounding onto Isat.
//   Nsat: fixed at a literature value, 1.8 (was fit in [0.1, 5.0]). The audit
//     found it weakly identified on its own (once N0 is comfortably above
//     Nsat, N/(N+Nsat) saturates and further N0/Nsat changes barely affect
//     the dynamics).
//   Z0: lower bound loosened 1e-5 -> 1e-8 (upper bound unchanged at 1e-3).
//     The audit found Z0 was simultaneously pinned at its old lower bound AND
//     the single stiffest (best-identified) direction in the whole spectrum
//     -- not an identifiability problem but a bounds problem: the data has a
//     genuine, sharp preference for Z0 smaller than 1e-5 allowed.
//
// Free parameters remaining: c0, N_supplement, N0, Z0, Z_max_ration, kmortP,
// Isat, background, sigma (9, down from 11). With this reparametrization the
// audit's best-found fit has NO parameters pinned at any bound (previously
// kprod and Z0 both were), and is the best loss found across every
// likelihood/parametrization variant tried so far this session.

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

  // Fixed at literature values per the identifiability audit -- see header.
  real kprod = 1.7;
  real Nsat = 1.8;

  // Same +P_min floor used on the predicted side, applied here so both sides
  // of the likelihood are on a consistent, strictly-positive scale (handles
  // the dataset's near-zero/negative fluorescence readings).
  vector[N_obs] log_Chla_obs = log(Chla_obs + P_min);
}

parameters {
  real<lower=0.5, upper=5> c0;
  real<lower=0.05, upper=2> N_supplement;
  real<lower=5, upper=40> N0;
  real<lower=1e-8, upper=0.001> Z0;  // lower bound loosened 1e-5 -> 1e-8, see header
  real<lower=0.1, upper=4.0> Z_max_ration;
  real<lower=0.01, upper=0.5> kmortP;
  real<lower=1.0, upper=80> Isat;
  real<lower=0.5, upper=15.0> background;
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
  // remaining parameters: implicit uniform prior over their declared bounds.

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
