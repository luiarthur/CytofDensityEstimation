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
from scipy.stats import ks_2samp

import pystan_util
import simulate_data
import posterior_inference

from importlib import reload
reload(simulate_data); reload(pystan_util); reload(posterior_inference)

import sys
sys.path.append('../util')
import util

# Simulate data.
def generate_scenarios(N, etaTK, p=1):
    eta_T = np.array([.5, 1 - etaTK - .5, etaTK])
    eta_T = np.clip(eta_T, 1e-16, 0.5)
    return simulate_data.gen_data(  # TODO: Juhee simulation.
        n_C=N, n_T=N, p=p, gamma_C=.3, gamma_T=.2, K=3,
        loc=np.array([-1, 1, 2]), scale=np.array([0.7, 1.3, 1.0]), 
        nu=np.array([15, 30, 10]), phi=np.array([-2, -5, 0]),
        # TODO: 
        #     - Loop through these, run multiple chains.
        #     - Select best run (seed). Report results.
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .50, 1e-16]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .45, .05]),
        eta_C=np.array([.5, .5, 1e-16]), eta_T=eta_T,
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .35, .15]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .30, .20]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .25, .25]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .20, .30]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .15, .35]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .10, .40]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, .05, .45]),
        # eta_C=np.array([.5, .5, 1e-16]), eta_T=np.array([.5, 1e-16, .5]),
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
                                             a_p=.5, b_p=.5)
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

    # Optimization.
    # NOTE: Needs to be run several times with different initial values.
    opt_fit = sm.optimizing(data=stan_data, tol_obj=0.5, seed=stan_seed)

    # simulate.
    tic = time.time()
    if method == 'advi':
        # NOTE: Seed is very important for ADVI.
        #       Need to run with different seeds and pick run with best ELBO. 
        _fit = sm.vb(data=stan_data, iter=2000, seed=stan_seed, pars=pars,
                     # diagnostic_file=f'{results_dir}/diagnostic.csv',
                     # sample_file=f'{results_dir}/samples.csv',
                     # NOTE: log_p__ in `sample_file` is the
                     # log unnormalized posterior.
                     output_samples=1000, algorithm='meanfield')
    else:
        get_val = lambda k: opt_fit[k].item() if opt_fit[k].size == 1 else opt_fit[k]
        init = dict([(k, get_val(k)) for k in opt_fit])
        print('Using initial values from penalized mle:')
        print(init)
        fit = sm.sampling(data=stan_data, init=[init], iter=2000,
                          warmup=1000, pars=pars, thin=1, chains=1,
                          seed=stan_seed)

    toc = time.time()
    print(f'Model inference time: {toc - tic}')

    if method == 'advi':
        fit = pystan_util.vb_extract(_fit)
        print(fit.keys())

    # Append beta to results.
    fit['beta'] = posterior_inference.beta_posterior(fit, stan_data)
    posterior_inference.print_stat('beta', fit)
    posterior_inference.print_summary(fit, truth=data, digits=3, show_sd=True,
                                      include_gamma_eta_T=True)

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

    # KS test.
    ks = ks_2samp(stan_data['y_T'], stan_data['y_C'])
    print(ks)

    return dict(data=data, fit=fit, ks=ks, opt_fit=opt_fit)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        results_dir = 'results/test/quick'
        etaTK = 0.1  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
        method = "advi" # advi,nuts
        stanseed = 1  # 1,2,3,4,5
    else:
        results_dir = sys.argv[1]
        etaTK = float(sys.argv[2])
        method = sys.argv[3]
        stanseed = int(sys.argv[4])

    os.makedirs(results_dir, exist_ok=True)

    # Compile model if needed.
    os.system('make compile')

    # Load model.
    sm = pickle.load(open(f'.model.pkl', 'rb'))

    # Scenarios:
    # - stan_seed = (1, 5) (for VB only)
    # - etaTK = (0, 0.05, 0.10, ..., 0.50)
    # - method = 'advi' or 'nuts'

    # Generate data.
    # data = generate_scenarios(p, N=200)
    data = generate_scenarios(N=400, etaTK=etaTK)  # larger means more different.

    # Plot simulation data.
    simulate_data.plot_data(yT=data['y_T'], yC=data['y_C'], bins=30)
    plt.savefig(f'{results_dir}/sim-data.pdf')
    plt.close()

    # Run analysis.
    results = simulation(data, p, method, results_dir, stan_seed=stanseed)
    with open(f'{results_dir}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Load with:
    # import pickle
    # results = pickle.load(open(f'{results_dir}/results.pkl', 'rb'))

