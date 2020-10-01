import re
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

# from importlib import reload; reload(simulate_data)
# from importlib import reload; reload(pystan_util)
# from importlib import reload; reload(posterior_inference)

import sys
sys.path.append('../util')
import util

# Simulate data.
def generate_scenarios(p):
    return simulate_data.gen_data(
        n_C=1000, n_T=1000, p=p, gamma_C=.3, gamma_T=.2, K=2,
        scale=np.array([0.7, 1.3]), nu=np.array([15, 30]),
        loc=np.array([1, -1]), phi=np.array([-2, -5]),
        eta_C=np.array([.99, .01]), eta_T=np.array([.01, .99]), seed=1)

def simulation(p, method, results_dir):
    # Stan data
    stan_data = pystan_util.create_stan_data(y_T=data['y_T'], y_C=data['y_C'], K=5,
                                             m_phi=-1, d_xi=0.31, d_phi=10,
                                             a_p=.1, b_p=.9, m_nu=3, s_nu=1)

    # simulate.
    tic = time.time()
    if method == 'advi':
        _fit = sm.vb(data=stan_data, iter=1000, seed=1,
                        grad_samples=2, elbo_samples=2, output_samples=1000)
    else:
        fit = sm.sampling(data=stan_data, iter=500, warmup=500,
                          thin=1, chains=1, seed=1)

    toc = time.time()
    print(f'Model inference time: {toc - tic}')

    if method == 'advi':
        fit = pystan_util.vb_extract(_fit)

    posterior_inference.print_summary(fit, truth=data, digits=3, show_sd=False)

    # Plot posterior predictive
    simulate_data.plot_data(stan_data['y_T'], stan_data['y_C'])
    y_grid = np.linspace(-8, 3, 200)
    posterior_inference.plot_post_predictive_density(fit, y_grid,
                                                     fill_alpha=0.6, mean_alpha=0)
    posterior_inference.plot_ci(fit['gamma_T'], loc=-10, alpha=.5,
                                color='red', lw=2)
    posterior_inference.plot_ci(fit['gamma_C'], loc=-10, alpha=.5,
                                color='blue', lw=2)
    plt.legend()
    plt.savefig(f'{results_dir}/postpred-simdata-p{p}-method{method}.pdf',
                bbox_inches='tight')
    plt.close()

def parse_p(results_dir):
    return float(re.findall(r'(?<=p_)\d+\.\d+', results_dir)[0])
    
def parse_method(results_dir):
    return re.findall(r'(?<=method_)\w+', results_dir)[0]
 

if __name__ == '__main__':
    if len(sys.argv) == 0:
        results_dir = 'results/test/p_0.95-method_advi'
    else:
        results_dir = sys.argv[0]
    
    p = parse_p(results_dir)
    method = parse_method(results_dir)

    # Compile model if needed.
    os.system('make compile')

    # Load model.
    sm = pickle.load(open(f'.model.pkl', 'rb'))

    # Scenarios:
    # - p = (0.95, 0.05)
    # - method = 'advi' or 'nuts'

    # Generate data.
    data = generate_scenarios(p)

    # Plot simulation data.
    simulate_data.plot_data(yT=data['y_T'], yC=data['y_C'], bins=30)
    plt.savefig(f'{results_dir}/sim-data-p{p}.pdf')
    plt.close()

    # Run analysis.
    simulation(p, method, results_dir)

