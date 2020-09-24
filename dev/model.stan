// Skew Normal distribution in STAN:
//   https://mc-stan.org/docs/2_24/functions-reference/skew-normal-distribution.html

functions {
  // nu:positive, loc:real, scale:positive, alpha:real
  real skew_t_lpdf(real x, real nu, real loc, real scale, real alpha) {
    real z;
    real u;
    real kernel;

    z = (x - loc) / scale;
    u = alpha * z * sqrt((nu + 1) / (nu + z*z));
    kernel = student_t_lpdf(z | nu, 0, 1) + student_t_lcdf(u | nu + 1, 0, 1);
    return (kernel + log(2) - log(scale));
  }
}

data {
  int<lower=0> N_T;
  int<lower=0> N_C;
  real y_T[N_T];
  real y_C[N_C];
  int<lower=0> K;  // number of mixture components.

  // hyper parameters
  real<lower=0> a_gamma;
  real<lower=0> b_gamma;

  vector<lower=0>[K] a_eta;

  real xi_bar;
  real<lower=0> d_xi;
  real<lower=0> d_phi;
  real<lower=0> a_sigma;
  real<lower=0> b_sigma;
  vector<lower=0>[K] nu;
}

transformed data {
  real x_T[N_T];
  real x_C[N_C];
  real spike_loc;
  real spike_sd;

  spike_loc = -10;
  spike_sd = 0.001;

  for (n in 1:N_T) {
    x_T[n] = is_inf(-y_T[n]) ? spike_loc : y_T[n];
  }

  for (n in 1:N_C) {
    x_C[n] = is_inf(-y_C[n]) ? spike_loc : y_C[n];
  }

  // int<lower=0, upper=1> not_expressed_T[N_T];
  // int<lower=0, upper=1> not_expressed_C[N_C];

  // for (n in 1:N_T) {
  //   not_expressed_T[n] = is_inf(y_T[n]);
  // }
  // for (n in 1:N_C) {
  //   not_expressed_T[n] = is_inf(y_C[n]);
  // }
}

parameters {
  real<lower=0, upper=1> gamma_tilde_T;
  real<lower=0, upper=1> gamma_C;

  simplex[K] eta_tilde_T;
  simplex[K] eta_C;

  vector[K] xi;
  vector[K] phi;
  vector<lower=0>[K] sigma;

  real<lower=0, upper=1> p;  // NOTE: Fixed in paper.
}

transformed parameters {
  real<lower=0, upper=1> gamma_T;
  // simplex[K] eta_T;  // simplex can't be guaranteed at compile time.
  vector[K] eta_T;  // technically a simplex.

  eta_T = p * eta_C + (1 - p) * eta_tilde_T;
  gamma_T = p * gamma_C + (1 - p) * gamma_tilde_T;
}

model {
  p ~ beta(1, 1);

  gamma_tilde_T ~ beta(a_gamma, b_gamma);
  gamma_C ~ beta(a_gamma, b_gamma);

  eta_tilde_T ~ dirichlet(a_eta);
  eta_C ~ dirichlet(a_eta);

  sigma ~ inv_gamma(a_sigma, b_sigma);
  xi ~ normal(xi_bar, d_xi * sigma);  // g-prior
  phi ~ normal(0, d_phi * sigma);  // g-prior
  
  {
    vector[N_T] lpdf_mix_T;
    vector[N_C] lpdf_mix_C;
    vector[K] lpdf_mix;
    for (n in 1:N_T) {
      for (k in 1:K) {
        lpdf_mix[k] = log(eta_T[k]) + skew_t_lpdf(x_T[n] | nu[k], xi[k], sigma[k], phi[k]);
        // lpdf_mix[k] = log(eta_T[k]) + normal_lpdf(x_T[n] | xi[k], sigma[k]);
      }
      lpdf_mix_T[n] = log_sum_exp(lpdf_mix);
    }

    for (n in 1:N_C) {
      for (k in 1:K) {
        lpdf_mix[k] = log(eta_C[k]) + skew_t_lpdf(x_C[n] | nu[k], xi[k], sigma[k], phi[k]);
        // lpdf_mix[k] = log(eta_C[k]) + normal_lpdf(x_C[n] | xi[k], sigma[k]);
      }
      lpdf_mix_C[n] = log_sum_exp(lpdf_mix);
    }

    for (n in 1:N_C) {
      // target += log_mix(gamma_C, log(is_zero_C[n]), lpdf_mix_C[n]);
      target += log_mix(gamma_C, normal_lpdf(x_C[n] | spike_loc, spike_sd), lpdf_mix_C[n]);
    }

    for (n in 1:N_T) {
      // target += log_mix(gamma_T, log(is_zero_T[n]), lpdf_mix_T[n]);
      target += log_mix(gamma_T, normal_lpdf(x_T[n] | spike_loc, spike_sd), lpdf_mix_T[n]);
    }
  }
}
