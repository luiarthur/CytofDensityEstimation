import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
dtype = np.float64
Root = tfd.JointDistributionCoroutine.Root

def compute_p0_T(gamma_C, gamma_T, p):
    return p * gamma_T + (1 - p) * gamma_C

def compute_pnot0_T(gamma_C, gamma_T, eta_C, eta_T, p):
    return eta_T * (1 - gamma_T) * p + eta_C * (1 - p) * (1 - gamma_C)

def compute_p0_C(gamma_C):
    return gamma_C

def compute_pnot0_C(gamma_C, eta_C):
    return gamma_C * eta_C;

def log_dirac_neginf(x, log_eps=-1000):
    return (x == -np.inf) * log_eps

# TODO: change to skew_t
def loglike_i(y, loc, scale):
    f = (tfd.Normal(loc[None, ...], scale[None, ...])
            .log_prob(np.clip(y, -100, np.inf)[:, None]))
    return f

# TODO: change to skew_t
def sample_dist_log_prob(y, loc, scale, p0, pnot0):
    f = tf.log(pnot0)[None, ...] + loglike_i(y, loc, scale)
    a = log_dirac_neginf(y) + tf.log(p0)
    return tf.reduce_logsumexp(fa.concat([f, a], axis=-1), axis=-1).sum()

def log_prob(y_C, y_T, gamma_C, gamma_T, eta_C, eta_T, p, loc, scale):
    p0_T = compute_p0_T(gamma_C, gamma_T, p)
    pnot0_T = compute_pnot0_T(gamma_C, gamma_T, eta_C, eta_T, p, gamma_T_star)
    loglike_i(y_C, y_C, 

