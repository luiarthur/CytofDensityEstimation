import stan2tfp
import numpy as np
import pandas as pd

def read_data(path, marker, subsample=None, random_state=None):
    donor = pd.read_csv(path, low_memory=True)
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

# Path to data.
data_dir = '../../data/TGFBR2/cytof-data'
path_to_donor1 = f'{data_dir}/donor1.csv'

# Read data.
# marker = 'NKG2D' # looks efficacious
marker = 'CD16'  # looks not efficacious

donor1_data = read_data(path_to_donor1, marker, subsample=2000, random_state=2)

stan_data = create_stan_data(y_T=donor1_data['y_T'], y_C=donor1_data['y_C'],
                             # K=5,  # old, works.
                             K=5, m_phi=-1, d_xi=100,  m_nu=4, s_nu=2, # testing
                             a_sigma=13, b_sigma=12)

with open('../model.stan', 'r') as f:
    stan_model = f.read()

out = stan2tfp.Stan2tfp(stan_file_path='../model.stan', data_dict=stan_data)

