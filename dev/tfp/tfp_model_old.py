import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
dtype = np.float64
Root = tfd.JointDistributionCoroutine.Root

def compute_gamma_T_star(gamma_C, gamma_T, p):
    return p * gamma_T + (1 - p) * gamma_C

def compute_eta_T_star(gamma_C, gamma_T, eta_C, eta_T, p, gamma_T_star):
    numer = eta_T * (1 - gamma_T) * p + eta_C * (1 - p) * (1 - gamma_C)
    return numer / (1 - gamma_T_star)


def mix_T(gamma_C, gamma_T, eta_C, eta_T, p, loc, scale, neg_inf, n):
    gamma_T_star = compute_gamma_T_star(gamma_C, gamma_T, p)
    eta_T_star = compute_eta_T_star(gamma_C, gamma_T, eta_C, eta_T, p, gamma_T_star)
    return mix(gamma_T_star, eta_T_star, loc, scale, neg_inf, n)


def mix(gamma, eta, loc, scale, neg_inf, n):
    return tfd.Mixture(
        cat=tfd.Categorical(probs=tf.stack([gamma, 1 - gamma], axis=-1)),
        components=[
            tfd.Sample(
                tfd.Normal(np.float64(neg_inf), 1e-5), sample_shape=n),
            tfd.Sample(
                tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=eta),
                components_distribution=tfd.Normal(loc=loc, scale=scale)),
                sample_shape=n)
        ])

# NOTE:
# - `Sample` and `Independent` resemble, respectively, `filldist` and `arraydist` in Turing.
def create_model(n_C, n_T, K, neg_inf=-10, dtype=np.float64):
    return tfd.JointDistributionNamed(dict(
        p = tfd.Beta(dtype(1), dtype(1)),
        gamma_C = tfd.Gamma(dtype(3), dtype(3)),
        gamma_T = tfd.Gamma(dtype(3), dtype(3)),
        eta_C = tfd.Dirichlet(tf.ones(K, dtype=dtype) / K),
        eta_T = tfd.Dirichlet(tf.ones(K, dtype=dtype) / K),
        loc = tfd.Sample(tfd.Normal(dtype(0), dtype(1)),
                         sample_shape=K),
        sigma_sq = tfd.Sample(tfd.InverseGamma(dtype(3), dtype(2)),
                              sample_shape=K),
        y_C = lambda gamma_C, eta_C, loc, sigma_sq: 
            mix(gamma_C, eta_C, loc, tf.sqrt(sigma_sq), dtype(neg_inf), n_C),
        y_T = lambda gamma_C, gamma_T, eta_C, eta_T, p, loc, sigma_sq: 
            mix_T(gamma_C, gamma_T, eta_C, eta_T, p, loc, tf.sqrt(sigma_sq),
                  dtype(neg_inf), n_T)
    ))


# TEST
n_C=4
n_T=6
K=3
model = create_model(n_C=n_C, n_T=n_T, K=K)
s = model.sample(); s
model.log_prob(s)
# model.sample(2)  # FIXME

gamma = tfd.Beta(dtype(1), 1.).sample(2)
eta = tfd.Dirichlet(tf.ones(K, dtype=dtype) / K).sample(2)
m = mix(gamma, eta, tf.zeros(K, dtype=dtype), tf.ones(K, dtype=dtype), dtype(-10), n_C)
s = m.sample(3)
m.log_prob(s)

