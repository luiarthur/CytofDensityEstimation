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

def mix(gamma, eta, loc, scale, neg_inf, name):
    return tfd.Mixture(
        cat=tfd.Categorical(probs=np.array([gamma, 1 - gamma])),
        components=[
            tfd.Deterministic(np.float64(neg_inf)),
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=eta),
                components_distribution=tfd.Normal(loc=loc, scale=scale)
            )
        ],
    name=name)

# NOTE:
# - `Sample` and `Independent` resemble, respectively, `filldist` and `arraydist` in Turing.
# - JointDistributionCoroutine is good in this case,
#   otherwise, I usually use JointDistributionNamed.
def create_model(K, neg_inf=-100, dtype=np.float64):
    def _model():
        p = yield Root(tfd.Beta(dtype(1), dtype(1), name="p"))
        gamma_C = yield Root(tfd.Gamma(dtype(1), dtype(1), name="gamma_C"))
        gamma_T = yield Root(tfd.Gamma(dtype(1), dtype(1), name="gamma_T"))
        eta_C = yield Root(tfd.Dirichlet(np.ones(K, dtype=dtype) / K, name="eta_C"))
        eta_T = yield Root(tfd.Dirichlet(np.ones(K, dtype=dtype) / K, name="eta_T"))
        loc = yield Root(tfd.Sample(tfd.Normal(dtype(0), dtype(1)), sample_shape=K, name="loc"))
        sigma_sq = yield Root(tfd.Sample(tfd.InverseGamma(dtype(3), dtype(2)), sample_shape=K,
                              name="sigma_sq"))
        scale = np.sqrt(sigma_sq)
        y_C = yield mix(gamma_C, eta_C, loc, scale, neg_inf, name="y_C")
        # TODO:
        # - y_T
        # - implement skew-t distribution
        # - prior for skew, nu parameter
    return tfd.JointDistributionCoroutine(_model)

model = create_model(5)
model.sample()
model.sample(2)


m = mix(0.5, [.1, .3, .6], np.zeros(3), np.ones(3), -10, name="x")
m.sample(5)
