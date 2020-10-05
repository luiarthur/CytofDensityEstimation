from jax import random, lax
from jax.scipy.special import betainc
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from SkewT import SkewT

class Mix(dist.Distribution):
    support = constraints.real_vector

    def __init__(self, df, loc, scale, skew, weights):
        super(Mix, self).__init__(event_shape=(1, ))
        self.df = df
        self.loc = loc
        self.scale = scale
        self.skew = skew
        self.weights = weights

    def sample(self, key, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        return jnp.zeros(sample_shape + self.event_shape)

    def log_prob(self, y, axis=-1):
        ll_comp = SkewT(self.df, self.loc, self.scale, self.skew).log_prob(y)
        ll_prob = jnp.log(self.weights)
        return logsumexp(ll_comp + ll_prob, axis=axis)

# Mix(2,3,4,5,[1.0]).log_prob(2)
# SkewT(2,3,4,5).log_prob(2)
