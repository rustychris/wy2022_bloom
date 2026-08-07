// Diagnostic-only variant: identical to npz_model.stan (full 11 free params,
// corrected pred-denominator likelihood, P_min=5) except integration is
// Stan's adaptive ode_rk45_tol instead of fixed-step RK4. Built solely to
// isolate whether the near-total optimize() failures seen with the
// pred-denominator/P_min=5 likelihood are caused by the likelihood itself or
// by RK4's known fragility near the edge of its safe dt_target range. Not
// intended as a long-lived model file.

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
  vector[N_obs] conc;
  vector[N_obs] age_d;
  vector[N_obs] kRadA;
  vector[N_obs] IsatA;
  vector[N_obs] kRadB;
  vector[N_obs] IsatB;
  vector[N_obs] Chla_obs;

  real<lower=0> thresh;
  real<lower=0> P_min;
}

transformed data {
  real kmortZ = 0.2;
  real Z_efficiency = 0.3;
  real mortP_to_N_efficiency = 0.5;
  real alpha = 0.15;
  real Ivlev = 1.0;
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
  sigma ~ normal(0, 1);

  for (n in 1:N_obs) {
    real resid = (Chla_obs[n] - P_pred[n]) / (P_pred[n] + P_min);
    resid ~ normal(0, sigma);
  }
}

generated quantities {
  real loss = 0;
  for (n in 1:N_obs) {
    real resid = (Chla_obs[n] - P_pred[n]) / (Chla_obs[n] + P_min);
    loss += resid^2;
  }
  loss /= N_obs;
}
