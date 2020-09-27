import pandas as pd
import numpy as np
import pystan

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns

from pystan_util import pystan_vb_extract
from posterior_inference import post_pred

def rm_inf(x):
    return list(filter(lambda x: not np.isinf(x), x))

def replace_inf(x, z):
    x = x + 0
    x[np.isinf(x)] = z
    return x

def print_stat(param, fit):
    m, s = fit[param].mean(0), fit[param].std(0)
    print(f'{param}: mean={np.round(m, 3)}, sd={np.round(s, 3)}')


def inv_gamma_moment(m, s):
    v = s*s
    a = (m / s) ** 2 + 2
    b = m * (a - 1)
    return a, b

def read_data(path, marker, subsample=None, random_state=None):
    donor = pd.read_csv(path)
    if subsample is not None:
      donor = donor.sample(n=subsample, random_state=random_state)

    y_C = donor[marker][donor.treatment.isna()]
    y_T = donor[marker][donor.treatment.isna() == False]
    assert y_C.shape[0] + y_T.shape[0] == donor.shape[0]

    return dict(y_C=np.log(y_C).to_numpy(),
                y_T=np.log(y_T).to_numpy())

def create_stan_data(y_C, y_T, K, p=0.5, a_gamma=1, b_gamma=1, a_eta=None,
                     a_sigma=3, b_sigma=2,
                     m_phi=0, xi_bar=None, d_xi=0.31, d_phi=0.31,
                     m_nu=3, s_nu=0.5):
    if a_eta is None:
        a_eta = np.ones(K) / K

    if xi_bar is None:
        _y = np.concatenate([y_C, y_T])
        xi_bar = np.mean(_y[_y > -np.inf])

    return dict(N_T=y_T.shape[0],
                N_C=y_C.shape[0],
                y_T=y_T,
                y_C=y_C,
                K=K, p=p,
                a_gamma=a_gamma, b_gamma=b_gamma, m_phi=m_phi,
                a_eta=a_eta, xi_bar=xi_bar, d_xi=d_xi, d_phi=d_phi,
                a_sigma=a_sigma, b_sigma=b_sigma,
                m_nu=m_nu, s_nu=s_nu)


# Compile STAN model.
# sm = pystan.StanModel('model.stan')

# Path to data.
data_dir = '../data/TGFBR2/cytof-data'
path_to_donor1 = f'{data_dir}/donor1.csv'

# Read data.
# FIXME: Remove subsample after testing!
marker = 'NKG2D' # looks efficacious
# marker = 'CD16'  # looks not efficacious

donor1_data = read_data(path_to_donor1, marker, subsample=2000, random_state=2)
# donor1_data = read_data(path_to_donor1, marker)
stan_data = create_stan_data(y_T=donor1_data['y_T'], y_C=donor1_data['y_C'],
                             # K=5,  # old, works.
                             K=5, m_phi=-1, d_xi=100,  m_nu=4, s_nu=2, # testing
                             a_sigma=13, b_sigma=12)
stan_data['y_T'], stan_data['y_C']

# ADVI.
vb_fit = sm.vb(data=stan_data, iter=1000, seed=2,
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

vb_fit = pystan_vb_extract(vb_fit)
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
plt.hist(rm_inf(stan_data['y_T']), color='red', 
         histtype='stepfilled',
         bins=50, alpha=0.6, density=True, label='T: Data')
plt.hist(rm_inf(stan_data['y_C']), color='blue',
         histtype='stepfilled',
         bins=50, alpha=0.6, density=True, label='C: Data')
sns.kdeplot(rm_inf(z[:, 1]), label="T: postpred", color='red')
sns.kdeplot(rm_inf(z[:, 0]), label="C: postpred", color='blue')
plt.scatter(-10, np.isinf(stan_data['y_T']).mean(),
            s=100, color='r', alpha=0.6, label='T: prop. zeros')
plt.scatter(-10, np.isinf(stan_data['y_C']).mean(),
            s=100, color='b', alpha=0.6, label='C: prob. zeros')
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

