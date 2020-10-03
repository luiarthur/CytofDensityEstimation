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

from importlib import reload
reload(simulate_data); reload(pystan_util); reload(posterior_inference)

import sys
sys.path.append('../util')
import util

# Simulate data.
def generate_scenarios(p, N):
    return simulate_data.gen_data(  # TODO: Juhee simulation.
        n_C=N, n_T=N, p=p, gamma_C=.3, gamma_T=.2, K=3,
        loc=np.array([-1, 1, 2]), scale=np.array([0.7, 1.3, 1.0]), 
        nu=np.array([15, 30, 10]), phi=np.array([-2, -5, 0]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .45, .05]),
        eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .40, .10]),
        seed=1)
    # return simulate_data.gen_data(
    #     n_C=N, n_T=N, p=p, gamma_C=.3, gamma_T=.2, K=2,
    #     scale=np.array([0.7, 1.3]), nu=np.array([15, 30]),
    #     loc=np.array([1, -1]), phi=np.array([-2, -5]),
    #     eta_C=np.array([.99, .01]), eta_T=np.array([.01, .99]),
    #     seed=1)

def simulation(data, p, method, results_dir, stan_seed=1):
    # Stan data
    stan_data = pystan_util.create_stan_data(y_T=data['y_T'], y_C=data['y_C'],
                                             K=5, m_phi=-1, s_mu=2, s_phi=3,
                                             a_p=1, b_p=1)
                                             # a_p=.1, b_p=.9)

    # Plot prior predictive.
    y_grid = np.linspace(-10, 8, 500)
    simulate_data.plot_data(yT=data['y_T'], yC=data['y_C'], bins=30)
    posterior_inference.plot_prior_density(
        stan_data, y_grid, B=1000, fill_alpha=0.4, mean_alpha=0)
    plt.xlim(-10.5, 10)
    plt.legend(loc='upper right')
    plt.savefig(f'{results_dir}/priorpred-simdata.pdf',
                bbox_inches='tight')
    plt.close()


    # Parameters to store
    pars = ['gamma_T', 'gamma_C', 'eta_T', 'eta_C',
            'gamma_T_star', 'eta_T_star',
            'mu', 'phi', 'sigma', 'nu', 'p']
    
    # simulate.
    tic = time.time()
    if method == 'advi':
        _fit = sm.vb(data=stan_data, iter=2000, seed=stan_seed, pars=pars,
                     grad_samples=2, elbo_samples=2, output_samples=1000)
    else:
        fit = sm.sampling(data=stan_data, iter=2000, warmup=1000, pars=pars,
                          thin=1, chains=1, seed=stan_seed)

    toc = time.time()
    print(f'Model inference time: {toc - tic}')

    if method == 'advi':
        fit = pystan_util.vb_extract(_fit)
        print(fit.keys())

    # Append beta to results.
    fit['beta'] = posterior_inference.beta_posterior(fit, stan_data)
    posterior_inference.print_stat('beta', fit)


    posterior_inference.print_summary(fit, truth=data, digits=3, show_sd=False)

    # Plot data.
    simulate_data.plot_data(stan_data['y_T'], stan_data['y_C'])

    # Plot posterior predictive
    y_grid = np.linspace(-8, 6, 500)
    posterior_inference.plot_post_density(
        fit, y_grid, fill_alpha=0.4, mean_alpha=0, use_eta_T=False)
    plt.legend()
    plt.savefig(f'{results_dir}/postpred-simdata.pdf',
                bbox_inches='tight')
    plt.close()

    return dict(data=data, fit=fit)

def parse_p(results_dir):
    return float(re.findall(r'(?<=p_)\d+\.\d+', results_dir)[0])
    
def parse_method(results_dir):
    return re.findall(r'(?<=method_)\w+', results_dir)[0]
 

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        results_dir = 'results/test/p_1.0-method_advi'
        # results_dir = 'results/test/p_0.0-method_advi'
        # results_dir = 'results/test/p_1.0-method_nuts'
        # results_dir = 'results/test/p_0.0-method_nuts'
    else:
        results_dir = sys.argv[1]

    os.makedirs(results_dir, exist_ok=True)
    p = parse_p(results_dir)
    method = parse_method(results_dir)

    # Compile model if needed.
    os.system('make compile')

    # Load model.
    sm = pickle.load(open(f'.model.pkl', 'rb'))

    # Scenarios:
    # - p = (1, 0)
    # - method = 'advi' or 'nuts'

    # Generate data.
    data = generate_scenarios(p, N=200)
    # data = generate_scenarios(p, N=1000)

    # Plot simulation data.
    simulate_data.plot_data(yT=data['y_T'], yC=data['y_C'], bins=30)
    plt.savefig(f'{results_dir}/sim-data.pdf')
    plt.close()

    # Run analysis.
    results = simulation(data, p, method, results_dir, stan_seed=5)
    with open(f'{results_dir}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Load with:
    # import pickle
    # results = pickle.load(open(f'{results_dir}/results.pkl', 'rb'))
    
