data {
  int<lower=0> N_T;
  int<lower=0> N_C;
  real<lower=0> y_T[N_T];
  real<lower=0> y_C[N_C];
  int<lower=0> K;  // number of mixture components.
  real<lower=0, upper=1> p;  // NOTE: Make this random?

  // hyper parameters
  real<lower=0> a_gamma;
  real<lower=0> b_gamma;

  vector<lower=0>[K] a_eta;

  real xi_bar;
  real<lower=0> d_xi;
  real<lower=0> d_phi;
  real<lower=0> a_sigma;
  real<lower=0> b_sigma;
}

transformed data {
  int<lower=0, upper=1> is_zero_T[N_T];
  int<lower=0, upper=1> is_zero_C[N_C];

  for (n in 1:N_T) {
    is_zero_T[n] = 1 * (y_T[n] == 0);
  }
  for (n in 1:N_C) {
    is_zero_C[n] = 1 * (y_C[n] == 0);
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
}

transformed parameters {
  real<lower=0, upper=1> gamma_T;
  simplex[K] eta_T;

  vector[N_T] lpdf_mix_T;
  vector[N_C] lpdf_mix_C;
  
  eta_T = p * eta_C + (1 - p) * eta_tilde_T;
  gamma_T = p * eta_C + (1 - p) * gamma_tilde_T;

  for (n in 1:N_T) {
    lpdf_mix_T[n] = 0;
    for (k in 1:K) {
      lpdf_mix_T[n] += log(eta_T[k]) + skew_normal_lpdf(y_T[n] | xi[k], sigma[k], phi[k]);
    }
  }

  for (n in 1:N_C) {
    lpdf_mix_C[n] = 0;
    for (k in 1:K) {
      lpdf_mix_C[n] += log(eta_C[k]) + skew_normal_lpdf(y_C[n] | xi[k], sigma[k], phi[k]);
    }
  }
}

model {
  gamma_tilde_T ~ beta(a_gamma, b_gamma);
  gamma_C ~ beta(a_gamma, b_gamma);
  eta_tilde_T ~ dirichlet(a_eta);
  eta_C ~ dirichlet(a_eta);

  sigma ~ inv_gamma(a_sigma, b_sigma);
  // TODO: Can I vectorize these?
  for (k in 1:K) {
    xi[k] ~ normal(xi_bar, d_xi * sigma[k]);  // g-prior
    phi[k] ~ normal(0, d_phi * sigma[k]);  // g-prior
  }
  
  for (n in 1:N_C) {
    target += log_mix(gamma_C, is_zero_C[n], log_sum_exp(lpdf_mix_C));
  }

  for (n in 1:N_T) {
    target += log_mix(gamma_T, is_zero_T[n], log_sum_exp(lpdf_mix_T));
  }
}
