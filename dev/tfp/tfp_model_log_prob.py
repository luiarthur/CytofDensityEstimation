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

def log_dirac_neginf(x, neginf=-np.inf, log_eps=-1000):
    return (x == neginf) * log_eps

# TODO: change to skew_t
def loglike_i(y, loc, scale):
    f = (tfd.Normal(loc[None, ...], scale[None, ...])
            .log_prob(np.clip(y, -100, np.inf)[:, None]))
    return f

# TODO: change to skew_t
def sample_dist_log_prob(y, loc, scale, gamma, eta):
    tf.log(eta) + loglike_i(y, loc, scale)

def log_prob(y_C, y_T, gamma_C, gamma_T, eta_C, eta_T, p, loc, scale):
    pass

