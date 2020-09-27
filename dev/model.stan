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

  real safe_skew_t_lpdf(real x, real nu, real loc, real scale, real alpha) {
    real neg_inf_approx = -10000;
    return is_inf(x) ? neg_inf_approx : skew_t_lpdf(x | nu, loc, scale, alpha);
  }

  real log_is_inf(real x) {
    return is_inf(x) ? 0 : negative_infinity();
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
  real m_phi;
  real<lower=0> d_xi;
  real<lower=0> d_phi;
  real<lower=0> a_sigma;
  real<lower=0> b_sigma;
  real m_nu;
  real<lower=0> s_nu;
}

parameters {
  real<lower=0, upper=1> gamma_T;
  real<lower=0, upper=1> gamma_C;

  simplex[K] eta_T;
  simplex[K] eta_C;

  vector[K] xi;
  vector[K] phi;
  vector<lower=0>[K] sigma_sq;
  vector<lower=0>[K] nu;

  real<lower=0, upper=1> p;  // NOTE: Fixed in paper.
}

transformed parameters {
  vector<lower=0>[K] sigma;
  real gamma_T_star;
  vector[K] eta_T_star;
  vector[K] eta_C_star;

  sigma = sqrt(sigma_sq);
  gamma_T_star = p * gamma_T + (1 - p) * gamma_C;
  eta_T_star = eta_T * p * (1 - gamma_T) + eta_C * (1 - p) * (1 - gamma_C);
  eta_C_star = eta_C * (1 - gamma_C);
}

model {
  p ~ beta(1, 1);  // prob. treatment has effect.

  gamma_T ~ beta(a_gamma, b_gamma);
  gamma_C ~ beta(a_gamma, b_gamma);

  eta_T ~ dirichlet(a_eta);
  eta_C ~ dirichlet(a_eta);

  sigma_sq ~ inv_gamma(a_sigma, b_sigma);
  xi ~ normal(xi_bar, d_xi * sigma);  // g-prior
  phi ~ normal(m_phi, d_phi * sigma);  // g-prior
  nu ~ lognormal(m_nu, s_nu);  // degrees of freedom
  
  {
    vector[K + 1] lpdf_mix;

    for (n in 1:N_C) {
      lpdf_mix[1:K] = log(eta_C_star);
      for (k in 1:K) {
        lpdf_mix[k] += safe_skew_t_lpdf(y_C[n] | nu[k], xi[k], sigma[k], phi[k]);
      }
      lpdf_mix[K + 1] = log(gamma_C) + log_is_inf(y_C[n]);
      target += log_sum_exp(lpdf_mix);
    }

    for (n in 1:N_T) {
      lpdf_mix[1:K] = log(eta_T_star);
      for (k in 1:K) {
        lpdf_mix[k] += safe_skew_t_lpdf(y_T[n] | nu[k], xi[k], sigma[k], phi[k]);
      }
      lpdf_mix[K + 1] = log(gamma_T_star) + log_is_inf(y_T[n]);
      target += log_sum_exp(lpdf_mix);
    }
  }
}
