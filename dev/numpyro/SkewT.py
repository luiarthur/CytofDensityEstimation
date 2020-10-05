from jax import random, lax
from jax.scipy.special import betainc
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

def std_studentt_cdf(z, df):
    x_t = df / (z ** 2 + df)
    neg_cdf = 0.5 * betainc(0.5 * df, 0.5, x_t)
    return jnp.where(z < 0., neg_cdf, 1. - neg_cdf)

def std_studentt_lcdf(z, df):
    return jnp.log(std_studentt_cdf(z, df))

# NOTE: Test the same.
# from tensorflow_probability import distributions as tfd
# tfd.StudentT(5, 0, 1).cdf(1)
# std_studentt_cdf(1,5)

# TODO: Are the shapes right???
#       See this: https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/continuous.py
class SkewT(dist.Distribution):
    support = constraints.real

    def __init__(self, df, loc, scale, skew):
        super(SkewT, self).__init__(event_shape=(1, ))
        self.df = df
        self.loc = loc
        self.scale = scale
        self.skew = skew

    def sample(self, key, sample_shape=()):
        # TODO.
        # it is enough to return an arbitrary sample with correct shape
        # return jnp.zeros(sample_shape + self.event_shape)
        key_gamma, key_tn, key_normal = random.split(key, 3)

        k = self.df / 2
        w = random.gamma(key_gamma, k, sample_shape) / k
        z = (dist.TruncatedNormal(loc=0., scale=jnp.sqrt(1/w), low=0.0)
                 .sample(key_tn))
        delta = self.skew / jnp.sqrt(1 + self.skew ** 2)

        _loc = self.loc + self.scale * z * delta
        _scale = self.scale * jnp.sqrt(1 - delta ** 2)
        return random.normal(key_normal, sample_shape) * _scale + _loc

    def log_prob(self, x):
        z = (x - self.loc) / self.scale
        u = self.skew * z * jnp.sqrt((self.df + 1) / (self.df + z ** 2));
        kernel = (dist.StudentT(self.df).log_prob(z) + 
                  std_studentt_lcdf(z=u, df=self.df + 1))
        return kernel + jnp.log(2/self.scale)

# TEST:
# SkewT(2,3,4,-5).log_prob(-1)  # approx -2.3487
# rng_key = random.PRNGKey(0)
# st = SkewT(12,3,4,jnp.array([-5, 5]))
# st.sample(rng_key, (10000, 2)).mean(0)
# st.sample(rng_key, (10000, 2)).std(0)
# st.log_prob(-1)
