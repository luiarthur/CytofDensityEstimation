import seaborn as sns
from scipy.stats import truncnorm, t
import numpy as np
import matplotlib.pyplot as plt

def rm_inf(x):
    return list(filter(lambda x: not np.isinf(x), x))

def rand_skew_t(nu, loc, scale, phi, size=None):
    w = np.random.gamma(nu/2, 2/nu, size=size)
    z = truncnorm.rvs(0, np.inf, scale=np.sqrt(1/w), size=size)
    delta = phi / np.sqrt(1 + phi**2)
    return (loc + scale * z * delta + 
            scale * np.sqrt(1 - delta**2) * np.random.normal(size=size))

def skew_t_lpdf(x, nu, loc, scale, skew, clip_min=-100):
    z = (np.clip(x, clip_min, np.inf) - loc) / scale
    u = skew * z * np.sqrt((nu + 1) / (nu + z * z));
    kernel = (t.logpdf(z, nu, 0, 1) + 
              t.logcdf(u, nu + 1, 0, 1))
    return kernel + np.log(2/scale)

def skew_t_pdf(x, nu, loc, scale, skew, clip_min=-100):
    return np.exp(skew_t_lpdf(x, nu, loc, scale, skew, clip_min))

def sample(n, gamma, eta, loc, scale, nu, phi):
    K = eta.shape[0]

    n_zero = np.clip(np.random.binomial(n, gamma), 1, n - 1)
    n_nonzero = n - n_zero
    k = np.random.choice(K, size=n_nonzero, replace=True, p=eta)
    y_nonzero = rand_skew_t(nu[k], loc[k], scale[k], phi[k],
                            size=n_nonzero)
    return np.concatenate([y_nonzero, np.full(n_zero, -np.inf)], axis=0)

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


def plot_data(yT, yC, tcolor='red', ccolor='blue', bins=50, alpha=0.6,
              tlabel='T: Data', clabel='C: Data', zero_pos=-10, 
              t0label='T prop. zeros', c0label='C: prop. zeros', 
              use_hist=False, lw=2, ls=":"):
    if use_hist:
        plt.hist(rm_inf(yT), color=tcolor, histtype='stepfilled', bins=bins,
                 alpha=alpha, density=True, label=tlabel)
        plt.hist(rm_inf(yC), color=ccolor, histtype='stepfilled', bins=bins,
                 alpha=alpha, density=True, label=clabel)
    else:
        sns.kdeplot(rm_inf(yT), color=tcolor, label=tlabel, ls=ls, lw=lw)
        sns.kdeplot(rm_inf(yC), color=ccolor, label=clabel, ls=ls, lw=lw)

    plt.scatter(zero_pos, np.isinf(yT).mean(),
                s=100, color=tcolor, alpha=alpha, label=t0label)
    plt.scatter(zero_pos, np.isinf(yC).mean(),
                s=100, color=ccolor, alpha=alpha, label=c0label)
 
def gen_data(n_C, n_T, p, gamma_C, gamma_T, K, eta_C=None, eta_T=None, nu=None,
             loc=None, scale=None, phi=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    if eta_C is None:
        eta_C = np.random.dirichlet(np.ones(K) / K)

    if eta_T is None:
        eta_T = np.random.dirichlet(np.ones(K) / K)

    if loc is None:
        loc = np.random.normal(2, 1, K)

    if scale is None:
        scale = np.random.normal(1, .1, K)

    if nu is None:
        nu = np.random.normal(30, 2.5, K)

    if phi is None:
        phi = np.random.normal(-3, 1, K)

    stat_T_star = compute_stat_T_star(gamma_C, gamma_T, eta_C, eta_T, p)
    y_C = sample(n_C, gamma_C, eta_C, loc, scale, nu, phi)
    y_T = sample(n_T, stat_T_star['gamma'], stat_T_star['eta'],
                 loc, scale, nu, phi)

    return dict(y_C=y_C, y_T=y_T, p=p, gamma_C=gamma_C, gamma_T=gamma_T,
                eta_C=eta_C, eta_T=eta_T, xi=loc, sigma=scale, nu=nu, 
                phi=phi)


if __name__ == '__main__':
    print('Testing...')

    data = gen_data(11, 9, p=.6, gamma_C=.4, gamma_T=.6, K=4)

    print('Tests execute successfully.')
    
    x = rand_skew_t(30, 2, .5, -3, size=10000)
    plt.hist(x, bins=100, density=True)
    xx = np.linspace(-1, 4, 1000)
    plt.plot(xx, skew_t_pdf(xx, 30, 2, .5, -3))
    plt.show()
