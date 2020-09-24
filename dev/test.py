import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pystan


def read_data(path, marker, na_val=-10, subsample=None, random_state=None):
    donor = pd.read_csv(path)
    if subsample is not None:
      donor = donor.sample(n=subsample, random_state=random_state)

    y_C = donor[marker][donor.treatment.isna()]
    y_T = donor[marker][donor.treatment.isna() == False]
    assert y_C.shape[0] + y_T.shape[0] == donor.shape[0]

    return dict(y_C=np.log(y_C).replace(-np.inf, na_val).to_numpy(),
                y_T=np.log(y_T).replace(-np.inf, na_val).to_numpy())


def create_stan_data(y_C, y_T, K, p, a_gamma=1, b_gamma=1, a_eta=None,
                     na_val=-10, xi_bar=None, d_xi=0.31, d_phi=0.31,
                     a_sigma=3, b_sigma=2, nu=None, nu_k=1):
    if a_eta is None:
        a_eta = np.ones(K) / K

    if xi_bar is None:
        xi_bar = np.concatenate([y_C, y_T]).mean()

    if nu is None:
        nu = np.full(K, nu_k)

    return dict(N_T=y_T.shape[0],
                N_C=y_C.shape[0],
                y_T=y_T,
                y_C=y_C,
                K=K, p=p,
                na_val=na_val,
                a_gamma=a_gamma, b_gamma=b_gamma,
                a_eta=a_eta, xi_bar=xi_bar, d_xi=d_xi, d_phi=d_phi,
                a_sigma=a_sigma, b_sigma=b_sigma, nu=nu)


# Compile STAN model.
sm = pystan.StanModel('model.stan')

# Path to data.
data_dir = '../data/TGFBR2/cytof-data'
path_to_donor1 = f'{data_dir}/donor1.csv'

# Read data.
# FIXME: Remove subsample after testing!
na_val = -6
donor1_data = read_data(path_to_donor1, 'CD16', subsample=1000, random_state=1,
                        na_val=na_val)
stan_data = create_stan_data(y_T=donor1_data['y_T'], y_C=donor1_data['y_C'],
                             na_val=na_val, K=5, p=0.5, d_xi=0.1, d_phi=0.1)
stan_data['y_T'], stan_data['y_C']

# ADVI. FIXME?!
# vb_fit = sm.vb(data=stan_data, iter=100, seed=2)

# HMC. FIXME?!
hmc_fit = sm.sampling(data=stan_data, iter=500, warmup=400, thin=1, seed=1,
                      algorithm='HMC', chains=1,
                      control=dict(stepsize=0.05, int_time=1, adapt_engaged=False))

hmc_fit['p'].mean(), hmc_fit['p'].std()
hmc_fit['gamma_T'].mean(), hmc_fit['gamma_T'].std()
hmc_fit['gamma_C'].mean(), hmc_fit['gamma_C'].std()

plt.hist(list(filter(lambda x: np.isnan(x) == False, stan_data['y_T'])), 
         bins=30, alpha=0.6, density=True, label='T')
plt.hist(list(filter(lambda x: np.isnan(x) == False, stan_data['y_C'])), 
         bins=30, alpha=0.6, density=True, label='C')
plt.legend()
plt.savefig('bla.pdf', bbox_inches='tight')
plt.close()
