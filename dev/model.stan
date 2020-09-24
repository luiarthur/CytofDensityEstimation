// Skew Normal distribution in STAN:
//   https://mc-stan.org/docs/2_24/functions-reference/skew-normal-distribution.html

functions {
  // real skew_t_lpdf(real x, real nu, real loc, real scale, real skew) {
  //   real z;
  //   real c;
  //   real f;

  //   z = (x - loc) / scale;
  //   c =  2 / (skew + 1/skew);
  //   f = (z > 0) ? student_t_lpdf(z / skew | nu, 0, 1) : student_t_lpdf(z * skew | nu, 0, 1);

  //   return c * f;
  // }

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
  real na_val;

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
  int<lower=0, upper=1> is_zero_T[N_T];
  int<lower=0, upper=1> is_zero_C[N_C];

  for (n in 1:N_T) {
    is_zero_T[n] = (y_T[n] == na_val);
  }
  for (n in 1:N_C) {
    is_zero_C[n] = (y_C[n] == na_val);
  }
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
  // simplex[K] eta_T;  // can't be simplex because we are adding.
  vector[K] eta_T;  // technically a simplex.

  vector[N_T] lpdf_mix_T;
  vector[N_C] lpdf_mix_C;
  
  eta_T = p * eta_C + (1 - p) * eta_tilde_T;
  gamma_T = p * gamma_C + (1 - p) * gamma_tilde_T;

  {
    vector[K] lpdf_mix;
    for (n in 1:N_T) {
      for (k in 1:K) {
        lpdf_mix[k] = log(eta_T[k]) + skew_t_lpdf(y_T[n] | nu[k], xi[k], sigma[k], phi[k]);
        // lpdf_mix[k] = log(eta_T[k]) + skew_normal_lpdf(y_T[n] | xi[k], sigma[k], phi[k]);
        // lpdf_mix[k] = log(eta_T[k]) + normal_lpdf(y_T[n] | xi[k], sigma[k]);
      }
      lpdf_mix_T[n] = log_sum_exp(lpdf_mix);
    }

    for (n in 1:N_C) {
      for (k in 1:K) {
        lpdf_mix[k] = log(eta_C[k]) + skew_t_lpdf(y_C[n] | nu[k], xi[k], sigma[k], phi[k]);
        // lpdf_mix[k] = log(eta_C[k]) + skew_normal_lpdf(y_C[n] | xi[k], sigma[k], phi[k]);
        // lpdf_mix[k] = log(eta_C[k]) + normal_lpdf(y_C[n] | xi[k], sigma[k]);
      }
      lpdf_mix_C[n] = log_sum_exp(lpdf_mix);
    }
  }
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
  
  for (n in 1:N_C) {
    // target += log_mix(gamma_C, log(is_zero_C[n]), lpdf_mix_C[n]);
    target += log_mix(gamma_C, normal_lpdf(y_C[n] | na_val, .0001), lpdf_mix_C[n]);
  }

  for (n in 1:N_T) {
    // target += log_mix(gamma_T, log(is_zero_T[n]), lpdf_mix_T[n]);
    target += log_mix(gamma_T, normal_lpdf(y_T[n] | na_val, .0001), lpdf_mix_T[n]);
  }
}
