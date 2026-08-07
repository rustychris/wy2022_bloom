// Phase 1 variant v2: some parameters fixed at constants, based on the Phase 1
// optimize() MAP estimate (see bloom_lagrangian_zoop_etal_stan_20260806.py's
// run_optimize() output).
//
// An earlier version of this file also narrowed the bounds on c0, N0, Z0, and
// kmortP; that made optimize()/sample() pin 4 of 6 free parameters at their new
// bounds (worse loss: 0.569 vs. npz_model.stan's 0.353), so those four have been
// reverted to npz_model.stan's original ranges here. Remaining changes vs.
// npz_model.stan:
//   N_supplement: fixed constant at its MAP estimate, 0.532846 (was fit in [0.05, 2])
//   Isat:         fixed constant at its MAP estimate, 1.0001 (was fit in [1.0, 80])
//   background:   fixed constant at its MAP estimate, 1.57391 (was fit in [0.5, 15.0])
//   Nsat:         fixed constant at its MAP estimate, 3.35301 (was fit in [0.1, 5.0])
//   sigma:        fixed constant at its npz_model.stan MAP estimate, 0.594245
//                 (was fit with a normal(0,1) half-normal prior)
//
// With all four fixed exactly at the joint npz_model.stan MAP estimate, the
// original optimum is a true stationary point of this reduced problem: starting
// optimize() there reproduces loss=0.3518 exactly (verified). See how it fares
// from other starting points below via --multistart.
//
// See npz_model.stan for the base model, integration approach (fixed-step RK4
// with a clamped kLight -- same rationale/design notes apply here), and other
// design notes -- unchanged here.

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

  // state y = [N, P, Z]. Autonomous (no explicit t-dependence).
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
  real<lower=0> dt_target; // fixed-step RK4 target step size, days
}

transformed data {
  // Parameters held fixed at NPZIVModel.base_params defaults for this phase --
  // not part of the original "zooplankton first go" tuned parameter set.
  real kmortZ = 0.2;
  real Z_efficiency = 0.3;
  real mortP_to_N_efficiency = 0.5;
  real alpha = 0.15;
  real Ivlev = 1.0;

  // Fixed at Phase 1 (npz_model.stan) optimize() MAP estimates.
  real N_supplement = 0.532846;
  real Isat = 1.0001;
  real background = 1.57391;
  real Nsat = 3.35301;
  real sigma = 0.594245;

  array[N_obs] int<lower=1> n_steps;
  for (n in 1:N_obs) {
    n_steps[n] = age_d[n] > 0 ? max(1, to_int(ceil(age_d[n] / dt_target))) : 1;
  }
}

parameters {
  real<lower=0.5, upper=5> c0;             // reverted to npz_model.stan's range
  real<lower=0.5, upper=2.0> kprod;
  real<lower=5, upper=40> N0;              // reverted to npz_model.stan's range
  real<lower=0.00001, upper=0.001> Z0;     // reverted to npz_model.stan's range
  real<lower=0.1, upper=4.0> Z_max_ration;
  real<lower=0.01, upper=0.5> kmortP;      // reverted to npz_model.stan's range
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
  // remaining parameters: implicit uniform prior over their declared bounds,
  // mirroring the original scipy bounds used with method='powell'. (sigma is
  // now a fixed constant, not a parameter, so it needs no prior statement.)

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
