import pandas as pd
import numpy as np
import pystan

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns

from pystan_util import pystan_vb_extract, create_stan_data

import simulate_data
import posterior_inference

import sys
sys.path.append('../util')
import util

def print_stat(param, fit):
    m, s = fit[param].mean(0), fit[param].std(0)
    print(f'{param}: mean={np.round(m, 3)}, sd={np.round(s, 3)}')

def inv_gamma_moment(m, s):
    v = s*s
    a = (m / s) ** 2 + 2
    b = m * (a - 1)
    return a, b


# Compile STAN model.
sm = pystan.StanModel('model.stan')

# Path to data.
data_dir = '../../data/TGFBR2/cytof-data'
path_to_donor1 = f'{data_dir}/donor1.csv'

# Read data.
marker = 'NKG2D' # looks efficacious
# marker = 'CD16'  # looks not efficacious
# marker = 'EOMES'  # ?? looks efficacious
# marker = 'CD56'  # looks not efficacious (data truncated above 0?)

# TODO: Remove subsample after testing!
donor1_data = util.read_data(path_to_donor1, marker, subsample=2000, random_state=2)
# donor1_data = read_data(path_to_donor1, marker)

stan_data = create_stan_data(y_T=donor1_data['y_T'], y_C=donor1_data['y_C'],
                             # K=5,  # old, works.
                             K=5, m_phi=-1, d_xi=100,  m_nu=4, s_nu=2, # testing
                             a_sigma=13, b_sigma=12)
stan_data['y_T'], stan_data['y_C']

# ADVI.
_vb_fit = sm.vb(data=stan_data, iter=1000, seed=2,
                grad_samples=1, elbo_samples=1, output_samples=2000)


# HMC.
# hmc_fit = sm.sampling(data=stan_data, 
#                       iter=500, warmup=400, thin=1, seed=1,
#                       algorithm='HMC', chains=1,
#                       control=dict(stepsize=0.01, int_time=1, adapt_engaged=False))

# NUTS.
# nuts_fit = sm.sampling(data=stan_data, 
#                        iter=500, warmup=400, thin=1, seed=1, chains=1)
# 

vb_fit = vb_extract(_vb_fit)
def doit():
    print_stat('p', vb_fit)
    print_stat('gamma_T', vb_fit)
    print_stat('gamma_C', vb_fit)
    print(f"T0: {np.isinf(stan_data['y_T']).mean()}")
    print(f"C0: {np.isinf(stan_data['y_C']).mean()}")
    print_stat('sigma', vb_fit)
    print_stat('phi', vb_fit)
    print_stat('xi', vb_fit)
    print_stat('nu', vb_fit)
    print_stat('eta_T', vb_fit)
    print_stat('eta_C', vb_fit)

doit()

# plot log prob for HMC/NUTS
plt.plot(vb_fit['lp__'])
plt.savefig('img/log_prob.pdf', bbox_inches='tight')
plt.close()

# plot posterior predictive
import importlib; importlib.reload(posterior_inference); importlib.reload(simulate_data)
simulate_data.plot_data(stan_data['y_T'], stan_data['y_C'])
y_grid = np.linspace(-8, 8, 200)
posterior_inference.plot_post_predictive_density(vb_fit, y_grid, fill_alpha=0)
plt.legend()
plt.savefig('img/postpred.pdf', bbox_inches='tight')
plt.close()

# Plot posterior p, gamma
plt.figure()
plt.subplot(1, 2, 1)
plt.hist(vb_fit['p'], density=True, bins=50)
plt.title(f"mean: {np.round(np.mean(vb_fit['p']), 3)}")
plt.xlabel(f'prob. of treatment effect \n for marker {marker}')
plt.ylabel('density')
plt.subplot(1, 2, 2)
plt.boxplot(np.vstack([vb_fit['gamma_T'], vb_fit['gamma_C']]).T)
plt.xticks(np.arange(2) + 1, [r'$\gamma_T$', r'$\gamma_C$'])
plt.scatter(1, np.isinf(stan_data['y_T']).mean(),
            s=100, color='r', alpha=0.6, label='T: prop. zeros')
plt.scatter(2, np.isinf(stan_data['y_C']).mean(),
            s=100, color='b', alpha=0.6, label='C: prob. zeros')
plt.legend()
plt.savefig('img/p.pdf', bbox_inches='tight')
plt.close()

