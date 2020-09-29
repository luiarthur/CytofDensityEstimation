import numpy as np
from posterior_inference import rand_skew_t

def sample(n, gamma, eta, loc, scale, nu, alpha):
    K = eta.shape[0]

    n_zero = np.clip(np.random.binomial(n, gamma), 1, n - 1)
    n_nonzero = n - n_zero
    k = np.random.choice(K, size=n_nonzero, replace=True, p=eta)
    y_nonzero = rand_skew_t(nu[k], loc[k], scale[k], alpha[k],
                            size=n_nonzero)
    return np.concatenate([y_nonzero, np.zeros(n_zero)], axis=0)

def compute_gamma_T_star(gamma_C, gamma_T, p):
    return p * gamma_T + (1 - p) * gamma_C

def compute_eta_T_star(gamma_C, gamma_T, eta_C, eta_T, p, gamma_T_star):
    numer = eta_T * (1 - gamma_T) * p + eta_C * (1 - p) * (1 - gamma_C)
    return numer / (1 - gamma_T_star)

def compute_stat_T_star(gamma_C, gamma_T, eta_C, eta_T, p):
    gamma_T_star = compute_gamma_T_star(gamma_C, gamma_T, p)
    eta_T_star = compute_eta_T_star(gamma_C, gamma_T, eta_C, eta_T, p,
                                    gamma_T_star)
    return dict(gamma=gamma_T_star, eta=eta_T_star)


if __name__ == '__main__':
    print('Testing...')

    np.random.seed(0)
    n_C = 11
    n_T = 9
    p = .6
    gamma_C = .4
    gamma_T = .6
    K = 4
    eta_C = np.random.dirichlet(np.ones(K))
    eta_T = np.random.dirichlet(np.ones(K))
    loc = np.random.randn(K)
    scale = np.random.rand(K)
    nu = np.random.rand(K) * 30
    alpha = np.random.randn(K) * 3


    stat_T_star = compute_stat_T_star(gamma_C, gamma_T, eta_C, eta_T, p)
    yC = sample(n_C, gamma_C, eta_C, loc, scale, nu, alpha)
    yT = sample(n_T, stat_T_star['gamma'], stat_T_star['eta'], loc, scale, nu, alpha)

    print('Tests execute successfully.')
