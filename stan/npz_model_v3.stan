// Phase 1 variant v3: fixes sigma and Nsat at their MAP estimates from the
// CORRECTED model's own optimize() run (npz_model.stan with the kLight clamp,
// loss=0.3557): sigma=0.597782, Nsat=0.728789. Free parameters remaining:
// c0, kprod, N_supplement, N0, Z0, Z_max_ration, kmortP, Isat, background
// (9, down from 11).
//
// Integration: reverted to Stan's adaptive ode_rk45_tol (bounded rel_tol/
// abs_tol/max_num_steps), NOT the fixed-step RK4 used in npz_model.stan --
// the RK4 switch turned out not to help sample() wall-clock here (same
// stalls at max_treedepth=6 as the adaptive solver, ~12min vs adaptive's
// ~2.5min for the 11-param model) since the real bottleneck is NUTS needing
// deep trees for this posterior's geometry, not integrator cost per se. The
// kLight correctness fix (clamped to <=1) is kept regardless of integrator.

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
  real<lower=0> P_min;     // normalization floor in the residual (2.0, matches loss())
}

transformed data {
  // Parameters held fixed at NPZIVModel.base_params defaults for this phase --
  // not part of the original "zooplankton first go" tuned parameter set.
  real kmortZ = 0.2;
  real Z_efficiency = 0.3;
  real mortP_to_N_efficiency = 0.5;
  real alpha = 0.15;
  real Ivlev = 1.0;

  // Fixed at the CORRECTED model's (npz_model.stan, post-kLight-fix) own MAP
  // estimate, loss=0.3557.
  real sigma = 0.597782;
  real Nsat = 0.728789;
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
}

transformed parameters {
  vector[N_obs] P_pred;

  for (n in 1:N_obs) {
    real kLight = klight_interp(Isat, kRadA[n], IsatA[n], kRadB[n], IsatB[n],
                                 conc[n], age_d[n], thresh);
    vector[3] y0 = [N0, c0 * conc[n], Z0]';
    vector[3] state;

    if (age_d[n] > 0) {
      // Bounded rel_tol/abs_tol/max_num_steps: some HMC warmup proposals drive
      // the ODE into a stiff/runaway regime where unbounded ode_rk45 grinds
      // through huge internal step counts. Capping max_num_steps here makes
      // such a draw fail fast (a rejected proposal) instead of stalling the
      // sampler. (With the kLight clamp this is now a secondary safety net
      // rather than the primary defense.)
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
  // remaining parameters: implicit uniform prior over their declared bounds.
  // (sigma, Nsat are now fixed constants, no priors needed.)

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
