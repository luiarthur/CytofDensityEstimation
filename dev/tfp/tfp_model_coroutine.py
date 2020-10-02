import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
dtype = np.float64
Root = tfd.JointDistributionCoroutine.Root

import sys
sys.path.append("../pystan")
import simulate_data

def compute_gamma_T_star(gamma_C, gamma_T, p):
    return p * gamma_T + (1 - p) * gamma_C

def compute_eta_T_star(gamma_C, gamma_T, eta_C, eta_T, p, gamma_T_star):
    numer = eta_T * (1 - gamma_T) * p + eta_C * (1 - p) * (1 - gamma_C)
    return numer / (1 - gamma_T_star)

def mix(n, eta, loc, scale, name):
    return tfd.Sample(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=eta),
            components_distribution=tfd.Normal(loc=loc, scale=scale),
            name=name),
        sample_shape=n)

# NOTE:
# - `Sample` and `Independent` resemble, respectively, `filldist` and `arraydist` in Turing.
# - JointDistributionCoroutine is good in this case,
#   otherwise, I usually use JointDistributionNamed.
def create_model(nC, nT, n0C, n0T, K, m_phi=-1, s_phi=3, dtype=np.float64):
    nposC = nC - n0C
    nposT = nT - n0T

    def _model():
        p = yield Root(tfd.Beta(dtype(1), dtype(1), name="p"))
        gamma_C = yield Root(tfd.Beta(dtype(1), dtype(1), name="gamma_C"))
        gamma_T = yield Root(tfd.Beta(dtype(1), dtype(1), name="gamma_T"))
        eta_C = yield Root(tfd.Dirichlet(np.ones(K, dtype=dtype) / K,
                                         name="eta_C"))
        eta_T = yield Root(tfd.Dirichlet(np.ones(K, dtype=dtype) / K,
                                         name="eta_T"))
        loc = yield Root(tfd.Sample(tfd.Normal(dtype(0), dtype(1)),
                                    sample_shape=K, name="loc"))
        nu = yield Root(tfd.Sample(tfd.Uniform(dtype(10), dtype(50)),
                                   sample_shape=K, name="nu"))
        phi = yield Root(tfd.Sample(tfd.Normal(dtype(m_phi), dtype(s_phi)),
                                    sample_shape=K, name="phi"))
        sigma_sq = yield Root(tfd.Sample(tfd.InverseGamma(dtype(3), dtype(2)),
                                         sample_shape=K,
                              name="sigma_sq"))
        scale = np.sqrt(sigma_sq)

        gamma_T_star = compute_gamma_T_star(gamma_C, gamma_T, p)
        eta_T_star = compute_eta_T_star(gamma_C[..., tf.newaxis],
                                        gamma_T[..., tf.newaxis],
                                        eta_C, eta_T,
                                        p[..., tf.newaxis],
                                        gamma_T_star[..., tf.newaxis])

        # likelihood
        y_C = yield mix(nC, eta_C, loc, scale, name="y_C")
        n0C = yield tfd.Binomial(nC, gamma_C, name="n0C")
        y_T = yield mix(nT, eta_T_star, loc, scale, name="y_T")
        n0T = yield tfd.Binomial(nT, gamma_T_star, name="n0T")

        # TODO:
        # - y_T
        # - implement skew-t distribution
        # - prior for skew, nu parameter

    return tfd.JointDistributionCoroutine(_model)

# Test compile. 
model = create_model(9, 11, 3, 6, 5)
s = model.sample(); s
s = model.sample(10); s
model.log_prob(s)

# Bijectors.
bijector = [
    tfb.Sigmoid(),  # p
    tfb.Sigmoid(),  # gamma_C
    tfb.Sigmoid(),  # gamma_T
    tfb.SoftmaxCentered(),  # eta_C
    tfb.SoftmaxCentered(),  # eta_T
    tfb.Identity(),  # loc
    tfb.Exp()  # sigma_sq
]

data = simulate_data.gen_data(
    n_C=1000, n_T=1000, p=0.95, gamma_C=.3, gamma_T=.2, K=2,
    scale=np.array([0.7, 1.3]), nu=np.array([15, 30]),
    loc=np.array([1, -1]), phi=np.array([-2, -5]),
    eta_C=np.array([.99, .01]), eta_T=np.array([.01, .99]), seed=1)

# TODO:
# - Try to do HMC/NUTS/ADVI.
# - Need to implement skew-t.

K = 5
model = create_model(10, 11, 13, 15, K, dtype=np.float64)

