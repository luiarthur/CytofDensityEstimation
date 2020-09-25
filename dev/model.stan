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
  real neg_inf = -10000;
}

parameters {
  real<lower=0, upper=1> gamma_tilde_T;
  real<lower=0, upper=1> gamma_C;

  simplex[K] eta_tilde_T;
  simplex[K] eta_C;

  vector[K] xi;
  vector[K] phi;
  vector<lower=0>[K] sigma_sq;

  real<lower=0, upper=1> p;  // NOTE: Fixed in paper.
}

transformed parameters {
  vector<lower=0>[K] sigma;
  real<lower=0, upper=1> gamma_T;
  // simplex[K] eta_T;  // simplex can't be guaranteed at compile time.
  vector[K] eta_T;  // technically a simplex.

  sigma = sqrt(sigma_sq);
  eta_T = p * eta_C + (1 - p) * eta_tilde_T;
  gamma_T = p * gamma_C + (1 - p) * gamma_tilde_T;
}

model {
  p ~ beta(1, 1);  // prob. treatment has no effect.

  gamma_tilde_T ~ beta(a_gamma, b_gamma);
  gamma_C ~ beta(a_gamma, b_gamma);

  eta_tilde_T ~ dirichlet(a_eta);
  eta_C ~ dirichlet(a_eta);

  sigma_sq ~ inv_gamma(a_sigma, b_sigma);
  xi ~ normal(xi_bar, d_xi * sigma);  // g-prior
  phi ~ normal(0, d_phi * sigma);  // g-prior
  
  {
    vector[N_T] lpdf_mix_T;
    vector[N_C] lpdf_mix_C;
    vector[K] lpdf_mix;

    for (n in 1:N_T) {
      for (k in 1:K) {
        lpdf_mix[k] = log(eta_T[k]);
        lpdf_mix[k] += is_inf(y_T[n]) ? neg_inf : 
                       skew_t_lpdf(y_T[n] | nu[k], xi[k], sigma[k], phi[k]);
      }
      lpdf_mix_T[n] = log_sum_exp(lpdf_mix);
    }

    for (n in 1:N_C) {
      for (k in 1:K) {
        lpdf_mix[k] = log(eta_C[k]);
        lpdf_mix[k] += is_inf(y_C[n]) ? neg_inf : 
                       skew_t_lpdf(y_C[n] | nu[k], xi[k], sigma[k], phi[k]);
      }
      lpdf_mix_C[n] = log_sum_exp(lpdf_mix);
    }

    for (n in 1:N_C) {
      target += log_mix(gamma_C, log(is_inf(y_C[n])), lpdf_mix_C[n]);
    }

    for (n in 1:N_T) {
      target += log_mix(gamma_T, log(is_inf(y_T[n])), lpdf_mix_T[n]);
    }
  }
}

generated quantities {
  real<lower=0, upper=1> p_efficacious;
  p_efficacious = 1 - p;
}
