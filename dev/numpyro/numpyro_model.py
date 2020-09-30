import matplotlib.pyplot as plt
import numpy as np
from jax import random, lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer import SVI, ELBO
from numpyro.infer import MCMC, NUTS, HMC

import sys
sys.path.append("../util")
sys.path.append("../pystan")
import util, simulate_data

# Generate data.
data = simulate_data.gen_data(1000, 1000, p=.99, gamma_C=.3, gamma_T=.2, K=2, 
                              loc=np.array([1, -1]), phi=np.array([-2, -5]),
                              eta_C=np.array([.99, .01]), eta_T=np.array([.01, .99]),
                              seed=1)

# Plot simulation data.
simulate_data.plot_data(yT=data['y_T'], yC=data['y_C'], bins=30)
plt.savefig('img/sim-data.pdf')
plt.close()

def skew_t_lpdf(x, nu, loc, scale, skew, clip_min=-100):
    z = (jnp.clip(x, clip_min, jnp.inf) - loc) / scale
    u = skew * z * jnp.sqrt((nu + 1) / (nu + z * z));
    kernel = (dist.StudentT(nu).log_prob(z) + 
              dist.StudentT(nu + 1).log_cdf(u))  # not implemented...
    return kernel + jnp.log(2/scale)


# ModelLikelihood
class Likelihhood(dist.Distribution):
    support = constraints.real

    def __init__(self, mu, sigma, w):
        super(NormalMixture, self).__init__(event_shape=(1, ))
        self.mu = mu
        self.sigma = sigma
        self.w = w

    def sample(self, key, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        return np.zeros(sample_shape + self.event_shape)

    def log_prob(self, y, axis=-1):
        lp = dist.Normal(self.mu, self.sigma).log_prob(y) + np.log(self.w)
        return logsumexp(lp, axis=axis
