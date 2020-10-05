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
from SkewT import SkewT
from Mix import Mix

import sys
sys.path.append("../util")
sys.path.append("../pystan")
import util
import simulate_data

# Generate data.
data = simulate_data.gen_data(200, 200, p=1, gamma_C=.3, gamma_T=.2, K=2, 
                              loc=np.array([1, -1]), phi=np.array([-2, -5]),
                              eta_C=np.array([.99, .01]),
                              eta_T=np.array([.01, .99]),
                              seed=1)

# Plot simulation data.
simulate_data.plot_data(yT=data['y_T'], yC=data['y_C'], bins=30)
plt.savefig('img/sim-data.pdf')
plt.close()

def create_model(yT, yC, num_components):
    # Cosntants
    nC = yC.shape[0]
    nT = yT.shape[0]
    zC = jnp.isinf(yC).sum().item()
    zT = jnp.isinf(yT).sum().item()
    yT_finite = yT[jnp.isinf(yT) == False]
    yC_finite = yC_finite = yC[jnp.isinf(yC) == False]
    K = num_components
    
    p = numpyro.sample('p', dist.Beta(.5, .5))
    gammaC = numpyro.sample('gammaC', dist.Beta(1, 1))
    gammaT = numpyro.sample('gammaT', dist.Beta(1, 1))

    etaC = numpyro.sample('etaC', dist.Dirichlet(jnp.ones(K) / K))
    etaT = numpyro.sample('etaT', dist.Dirichlet(jnp.ones(K) / K))
    
    with numpyro.plate('mixutre_components', K):
        nu = numpyro.sample('nu', dist.LogNormal(3.5, 0.5))
        mu = numpyro.sample('mu', dist.Normal(0, 3))
        sigma = numpyro.sample('sigma', dist.LogNormal(0, .5))
        phi = numpyro.sample('phi', dist.Normal(0, 3))


    gammaT_star = simulate_data.compute_gamma_T_star(gammaC, gammaT, p)
    etaT_star = simulate_data.compute_eta_T_star(etaC, etaT, p, gammaC, gammaT,
                                                 gammaT_star)

    with numpyro.plate('y_C', nC - zC):
        numpyro.sample('finite_obs_C',
                       Mix(nu[None, :],
                           mu[None, :],
                           sigma[None, :],
                           phi[None, :],
                           etaC[None, :]), obs=yC_finite[:, None])

    with numpyro.plate('y_T', nT - zT):
        numpyro.sample('finite_obs_T',
                       Mix(nu[None, :],
                           mu[None, :],
                           sigma[None, :],
                           phi[None, :],
                           etaT_star[None, :]), obs=yT_finite[:, None])

    numpyro.sample('N_C', dist.Binomial(nC, gammaC), obs=zC)
    numpyro.sample('N_T', dist.Binomial(nT, gammaT_star), obs=zT)


# Set random seed for reproducibility.
rng_key = random.PRNGKey(0)
# Set up NUTS sampler.
kernel = NUTS(create_model, max_tree_depth=10, target_accept_prob=0.8)
nuts = MCMC(kernel, num_samples=500, num_warmup=500)
nuts.run(rng_key, yC=jnp.array(data['y_C']), yT=jnp.array(data['y_T']),
         num_components=5)
nuts_samples = get_posterior_samples(nuts)
