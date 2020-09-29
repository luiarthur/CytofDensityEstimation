import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
dtype = np.float64
Root = tfd.JointDistributionCoroutine.Root

def skew_t_lpdf(x, nu, loc, scale, skew, clip_min=-100):
    z = (tf.clip_by_value(x, clip_min, np.inf) - loc) / scale
    u = skew * z * tf.sqrt((nu + 1) / (nu + z * z));
    kernel = (tfd.StudentT(nu, 0, 1).log_prob(z) + 
              tfd.StudentT(nu + 1, 0, 1).log_cdf(u))
    return kernel + tf.math.log(2/scale)

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

def create_prior(K, a_p=1, b_p=1, a_gamma=1, b_gamma=1, m_loc=0, g_loc=0.1,
                 m_sigma=3, s_sigma=2, m_nu=0, s_nu=1, m_skew=0, g_skew=0.1,
                 dtype=np.float64):
    return tfd.JointDistributionNamed(dict(
        p = tfd.Beta(dtype(a_p), dtype(b_p)),
        gamma_C = tfd.Gamma(dtype(a_gamma), dtype(b_gamma)),
        gamma_T = tfd.Gamma(dtype(a_gamma), dtype(b_gamma)),
        eta_C = tfd.Dirichlet(tf.ones(K, dtype=dtype) / K),
        eta_T = tfd.Dirichlet(tf.ones(K, dtype=dtype) / K),
        nu = tfd.Sample(tfd.LogNormal(dtype(m_nu), s_nu),
                        sample_shape=K),
        sigma_sq = tfd.Sample(tfd.InverseGamma(dtype(m_sigma), dtype(s_sigma)),
                              sample_shape=K),
        loc = lambda sigma_sq: tfd.Independent(
            tfd.Normal(dtype(m_loc), g_loc * tf.sqrt(sigma_sq)),
            reinterpreted_batch_ndims=1),
        skew = lambda sigma_sq: tfd.Independent(
            tfd.Normal(dtype(m_skew), g_skew * tf.sqrt(sigma_sq)),
            reinterpreted_batch_ndims=1),
    ))

def likelihood_i(p0, pnot0, nu, scale, loc, skew, y):
    lpdf0 = tf.math.log(p0) + log_dirac_neginf(y)

    # tf.broadcast_to ?
    # tmp = y.reshape(list(y.shape) + [1] * len(nu.shape))
    # print(tmp.shape)
    lpdf = skew_t_lpdf(y[..., None],
                       nu[None, ...], loc[None, ...], 
                       scale[None, ...], skew[None, ...])
    lpdf += tf.math.log(pnot0)

    ll = tf.reduce_logsumexp(
        tf.concat([lpdf, lpdf0[..., tf.newaxis]], axis=-1),
        axis=-1)

    return tf.reduce_sum(ll, axis=-1)


prior = create_prior(4)
s = prior.sample()
prior.log_prob(s)

# FIXME: Seems like there's no easy way to generate multiple samples.
#        Hence, no way to use ADVI.
# s = prior.sample(4)
# prior.log_prob(s)

skew_t_lpdf(x=tfd.Normal(np.float64(0), 1).sample(s['nu'].shape), nu=s['nu'],
            loc=s['loc'], scale=tf.sqrt(s['sigma_sq']), skew=s['skew'])

ll = likelihood_i(s['p'], (1 - s['p']) * s['eta_C'], s['nu'],
                  tf.sqrt(s['sigma_sq']), s['loc'], s['skew'],
                  np.random.randn(7))

