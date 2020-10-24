// Performance:
// - Map-reduce:
//       https://mc-stan.org/docs/2_22/stan-users-guide/using-map-reduce.html
// - Hierarchical Normal instead of direct t likelihood:
//       https://mc-stan.org/docs/2_22/stan-users-guide/reparameterization-section.html
// - Hurdle model:
//       https://mc-stan.org/docs/2_22/stan-users-guide/zero-inflated-section.html

functions {
  // Counts the number of infinities in array x.
  int count_inf(real[] x) {
    int num_inf = 0;
    for (n in 1:size(x)) {
      if (is_inf(x[n])) num_inf += 1;
    }
    return num_inf;
  }

  // Collects only the finite values in x.
  real[] collect_finite(real[] x) {
    int pos = 0;
    int num_inf = count_inf(x);
    real x_finite[size(x) - num_inf];
    for (n in 1:size(x)) {
      if (!is_inf(x[n])) {  // i.e., if x[n] is finite
         pos += 1;
         x_finite[pos] = x[n];
      }
    }
    return x_finite;
  }

  // Skew-t log pdf.
  // nu:positive, loc:real, scale:positive, phi:real
  real skew_t_lpdf(real x, real nu, real loc, real scale, real phi) {
    real z;
    real u;
    real kernel;

    z = (x - loc) / scale;
    u = phi * z * sqrt((nu + 1) / (nu + z*z));
    kernel = student_t_lpdf(z | nu, 0, 1) + student_t_lcdf(u | nu + 1, 0, 1);

    return (kernel + log(2) - log(scale));
  }

  real loglike_neginf(int N_infinite, int N_finite, real gamma) {
    return N_infinite * log(gamma) + (N_finite) * log1m(gamma);
  }

  // Log likelihood for mixture.
  real loglike_finite(matrix F_finite, vector eta) {
    
    int N_finite = rows(F_finite);
    int K = rows(eta);
    vector[N_finite] res;
    vector[K] log_eta = log(eta);
    vector[K] lpdf_mix;

    for (n in 1:N_finite) {
      for (k in 1:K) {
        lpdf_mix[k] = F_finite[n, k] + log_eta[k];
      }
      res[n] = log_sum_exp(lpdf_mix);
    }

    // Vectorized for efficiency:
    // https://mc-stan.org/docs/2_22/stan-users-guide/vectorization.html
    return sum(res);
  }

  matrix skew_t_lpdf_hierarchical(real[] y_finite, vector zeta, vector v,
                                  vector loc, vector scale, vector nu, vector phi) {
    int N_finite = size(y_finite);
    int K = rows(loc);
    matrix[N_finite, K] F;
    vector[K] omega_root = scale ./ sqrt(1 + phi .* phi);
    vector[K] psi = phi .* omega_root;
    real loc_nk;
    real scale_nk;

    for (n in 1:N_finite) for (k in 1:K) {
      loc_nk = loc[k] + psi[k] * zeta[n];
      scale_nk = omega_root[k] / sqrt(v[n]);
      F[n, k] = normal_lpdf(y_finite[n] | loc_nk, scale_nk);
      F[n, k] += gamma_lpdf(v[n] | nu[k]/2, nu[k]/2);
    }

    return F;
  }

  real loglike(matrix F_finite, int N_infinite, vector eta, real gamma) {
    int N_finite = rows(F_finite);
    real ll;

    ll = loglike_neginf(N_infinite, N_finite, gamma);
    ll += loglike_finite(F_finite, eta);

    return ll;
  }
}

data {
  int<lower=0> N_T;  // Number of observations in sample T.
  int<lower=0> N_C; // Number of observations in sample C.
  real y_T[N_T];  // Sample T, the log-transformed data with support on entire real line.
  real y_C[N_C];  // Sample C, the log-transformed data with support on entire real line.
  int<lower=0> K;  // number of mixture components.

  // Hyper parameters for priors.
  real<lower=0> a_gamma;
  real<lower=0> b_gamma;
  real<lower=0> a_p;
  real<lower=0> b_p;
  vector<lower=0>[K] a_eta;
  real mu_bar;
  real m_phi;
  real<lower=0> s_mu;
  real<lower=0> s_phi;
  real<lower=0> a_sigma;
  real<lower=0> b_sigma;
  real<lower=0> m_nu;
  real<lower=0> s_nu;
}

transformed data {
  int<lower=0, upper=N_C> N_neginf_C = count_inf(y_C);  // number of -Inf in sample C
  int<lower=0, upper=N_C> N_finite_C = N_C - N_neginf_C;  // number of finite values in samples C
  real y_finite_C[N_finite_C] = collect_finite(y_C);  // only the finite values in sample C

  int<lower=0, upper=N_T> N_neginf_T = count_inf(y_T);  // number of -Inf in sample T
  int<lower=0, upper=N_T> N_finite_T = N_T - N_neginf_T;  // number of finite values in samples T
  real y_finite_T[N_finite_T] = collect_finite(y_T);  // only the finite values in sample T
}

parameters {
  real<lower=0, upper=1> gamma_T;
  real<lower=0, upper=1> gamma_C;

  simplex[K] eta_T;
  simplex[K] eta_C;

  real<lower=0, upper=1> p;

  // Hierarchical representation of skew-t.
  vector[K] mu;  // mixture locations. (Don't care about order.)
  vector[K] phi;
  vector<lower=0>[N_finite_C] v_C;
  vector<lower=0>[N_finite_T] v_T;
  vector<lower=0>[N_finite_C] zeta_C;
  vector<lower=0>[N_finite_T] zeta_T;
  vector<lower=0>[K] sigma_sq;  // mixture scales.
  vector<lower=0>[K] nu;  // mixture degrees of freedoms.
}

transformed parameters {
  vector<lower=0>[K] sigma = sqrt(sigma_sq);
}

model {
  matrix[N_finite_C, K] FC_finite;
  matrix[N_finite_T, K] FT_finite;

  p ~ beta(a_p, b_p);  // prob. treatment has effect.

  // Probability of zeros.
  gamma_T ~ beta(a_gamma, b_gamma);
  gamma_C ~ beta(a_gamma, b_gamma);

  // Mixture weights.
  eta_T ~ dirichlet(a_eta);
  eta_C ~ dirichlet(a_eta);

  // auxiliary parameters
  for (n in 1:N_finite_C) zeta_C[n] ~ normal(0, sqrt(1/v_C[n]))T[0, ];
  for (n in 1:N_finite_T) zeta_T[n] ~ normal(0, sqrt(1/v_T[n]))T[0, ];

  // Mixture parameters.
  mu ~ normal(mu_bar, s_mu);  // locations
  sigma_sq ~ inv_gamma(a_sigma, b_sigma);  // squared scale
  nu ~ lognormal(m_nu, s_nu);  // degrees of freedom
  phi ~ normal(m_phi, s_phi);  // skewness
  
  // Matrix of skew t lpdf evaluations.
  FC_finite = skew_t_lpdf_hierarchical(y_finite_C, zeta_C, v_C, mu, sigma, nu, phi);
  FT_finite = skew_t_lpdf_hierarchical(y_finite_T, zeta_T, v_T, mu, sigma, nu, phi);

  target += loglike(FC_finite, N_neginf_C, eta_C, gamma_C);
  target += log_mix(p,
                    loglike(FT_finite, N_neginf_T, eta_T, gamma_T),
                    loglike(FT_finite, N_neginf_T, eta_C, gamma_C));
}

// Doesn't work with ADVI.
generated quantities {
  real ll = 0;
  real beta;
  {
    matrix[N_finite_C, K] FC_finite = skew_t_lpdf_hierarchical(
      y_finite_C, zeta_C, v_C, mu, sigma, nu, phi);
    matrix[N_finite_T, K] FT_finite = skew_t_lpdf_hierarchical(
      y_finite_T, zeta_T, v_T, mu, sigma, nu, phi);

    real llC = loglike(FC_finite, N_neginf_C, eta_C, gamma_C);
    real llTT = loglike(FT_finite, N_neginf_T, eta_T, gamma_T);
    real llTC = loglike(FT_finite, N_neginf_T, eta_C, gamma_C);
    real llT = log_mix(p, llTT, llTC);
    real logit_prob_beta_is_1 = logit(p) + llTT - llTC;

    ll = llC + llT;
    beta = inv_logit(logit_prob_beta_is_1) > uniform_rng(0, 1);
  }
}
