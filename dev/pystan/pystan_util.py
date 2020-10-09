import pandas as pd
import re
import numpy as np
from collections import OrderedDict

def vb_extract(results):
    param_specs = results['sampler_param_names']
    samples = results['sampler_params']
    n = len(samples[0])

    # first pass, calculate the shape
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            # no +1 for shape calculation because pystan already returns
            # 1-based indexes for vb!
            idxs = [int(i) for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

    # create arrays
    params = OrderedDict([(name, np.nan * np.empty((n, ) + tuple(shape)))
                          for (name, shape) in param_shapes.items()])

    # second pass, set arrays
    for param_spec, param_samples in zip(param_specs, samples):
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            # -1 because pystan returns 1-based indexes for vb!
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        params[name][(..., ) + tuple(idxs)] = param_samples

    return params


def create_stan_data(y_C, y_T, K, a_gamma=1, b_gamma=1, a_eta=None,
                     m_nu=3.5, s_nu=0.5,
                     mu_bar=None, s_mu=1,
                     a_sigma=3, b_sigma=2,
                     m_phi=0, s_phi=3,
                     a_p=1, b_p=1):
    if a_eta is None:
        a_eta = np.ones(K) / K

    if mu_bar is None:
        _y = np.concatenate([y_C, y_T])
        mu_bar = np.mean(_y[_y > -np.inf])

    return dict(N_T=y_T.shape[0],
                N_C=y_C.shape[0],
                a_p=a_p, b_p=b_p,
                y_T=y_T,
                y_C=y_C,
                K=K,
                a_gamma=a_gamma, b_gamma=b_gamma, m_phi=m_phi,
                a_eta=a_eta, mu_bar=mu_bar, s_mu=s_mu, s_phi=s_phi,
                a_sigma=a_sigma, b_sigma=b_sigma,
                m_nu=m_nu, s_nu=s_nu)

def group_vectors(d, key):
    columns = d.keys()
    rgx = f'{key}\.\d+'
    res = re.findall(rgx, ','.join(columns))
    return sorted(res)

def sanitize_vectors(d, key):
    group = group_vectors(d, key)
    d[key] = np.stack([d[k] for k in group], axis=-1)
    for g in group:
        d.pop(g)

def read_samples_file(samples_file,
                      vec_params=['mu', 'sigma_sq', 'sigma','nu', 'phi',
                                  'eta_T', 'eta_C']):
    samples = pd.read_csv(samples_file, comment='#')
    samples = {key: samples[key].to_numpy() for key in samples.keys()}
    for param in vec_params:
        sanitize_vectors(samples, param)
    return samples

