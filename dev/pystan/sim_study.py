import time
import pandas as pd
import numpy as np
import pystan
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns
import os

import pystan_util

import simulate_data
import posterior_inference

import sys
sys.path.append('../util')
import util

# Compile model if needed.
os.system('make compile')

# Load model.
sm = pickle.load(open('img/model.pkl', 'rb'))

# Simulate data.

effecacious = True

if effecacious:
    print('Simulating data that looks different')
    # Data looks different
    data = simulate_data.gen_data(
        1000, 1000, p=.95, gamma_C=.3, gamma_T=.2, K=2, scale=np.array([0.7, 1.3]),
        nu=np.array([15, 30]), loc=np.array([1, -1]), phi=np.array([-2, -5]),
        eta_C=np.array([.99, .01]), eta_T=np.array([.01, .99]), seed=1)
else:
    print('Simulating data that looks similar')
    # Data looks similar
    data = simulate_data.gen_data(
        1000, 1000, p=.01, gamma_C=.3, gamma_T=.2, K=2, scale=np.array([0.7, 1.3]),
        nu=np.array([15, 30]), loc=np.array([1, -1]), phi=np.array([-2, -5]),
        eta_C=np.array([.99, .01]), eta_T=np.array([.01, .99]), seed=1)


# Plot simulation data.
simulate_data.plot_data(yT=data['y_T'], yC=data['y_C'], bins=30)
plt.savefig('img/sim-data.pdf')
plt.close()

# Stan data
# from importlib import reload; reload(pystan_util)
stan_data = pystan_util.create_stan_data(y_T=data['y_T'], y_C=data['y_C'], K=5,
                                         m_phi=-1, d_xi=0.31, d_phi=10,
                                         a_p=.1, b_p=.9, m_nu=3, s_nu=1)


# ADVI.
tic = time.time()
_vb_fit = sm.vb(data=stan_data, iter=1000, seed=1,
                grad_samples=2, elbo_samples=2, output_samples=1000)
toc = time.time()
print(f'Model inference time: {toc - tic}')
vb_fit = pystan_util.vb_extract(_vb_fit)
posterior_inference.print_summary(vb_fit, truth=data)

# Plot posterior predictive
# from importlib import reload; reload(posterior_inference)
simulate_data.plot_data(stan_data['y_T'], stan_data['y_C'])
y_grid = np.linspace(-8, 8, 200)
posterior_inference.plot_post_predictive_density(vb_fit, y_grid,
                                                 fill_alpha=0.6, mean_alpha=0)
posterior_inference.plot_ci(vb_fit['gamma_T'], loc=-10, alpha=.5, color='red', lw=2)
posterior_inference.plot_ci(vb_fit['gamma_C'], loc=-10, alpha=.5, color='blue', lw=2)
plt.legend()
plt.savefig('img/postpred-simdata.pdf', bbox_inches='tight')
plt.close()
