import seaborn as sns
from scipy.stats import truncnorm, t
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def inv_gamma_moment(m, s):
    v = s*s
    a = (m / s) ** 2 + 2
    b = m * (a - 1)
    return a, b

def rm_inf(x):
    return list(filter(lambda x: not np.isinf(x), x))

def rand_skew_t(nu, loc, scale, phi, size=None):
    w = np.random.gamma(nu/2, 2/nu, size=size)
    z = truncnorm.rvs(0, np.inf, scale=np.sqrt(1/w), size=size)
    delta = phi / np.sqrt(1 + phi**2)
    return (loc + scale * z * delta + 
            scale * np.sqrt(1 - delta**2) * np.random.normal(size=size))

def skew_t_lpdf(x, nu, loc, scale, skew):
    z = (x - loc) / scale
    u = skew * z * np.sqrt((nu + 1) / (nu + z * z));
    kernel = (t.logpdf(z, nu, 0, 1) + 
              t.logcdf(u, nu + 1, 0, 1))
    return kernel + np.log(2/scale)

def skew_t_pdf(x, nu, loc, scale, skew):
    return np.exp(skew_t_lpdf(x, nu, loc, scale, skew))

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

def compute_eta_T_star(eta_C, eta_T, p):
    return p * eta_T + (1 - p) * eta_C

def compute_stat_T_star(gamma_C, gamma_T, eta_C, eta_T, p):
    gamma_T_star = compute_gamma_T_star(gamma_C, gamma_T, p)
    eta_T_star = compute_eta_T_star(eta_C, eta_T, p)
    return dict(gamma=gamma_T_star, eta=eta_T_star)


def get_true_density(data, y_grid=None, grid_length=1000, tcolor='red',
                     ccolor='blue', tlabel='T: Data', clabel='C: Data', lw=2,
                     ls=":"):
    if y_grid is None:
        yC_finite = data['y_C'][data['y_C'] > -np.inf]
        yT_finite = data['y_T'][data['y_T'] > -np.inf]
        y_finite = np.concatenate([yC_finite, yT_finite])
        y_min, y_max = y_finite.min(), y_finite.max()
        y_grid = np.linspace(y_min, y_max, grid_length)


    stat_T_star = compute_stat_T_star(data['gamma_C'], data['gamma_T'],
                                      data['eta_C'], data['eta_T'], data['p'])
    eta_T_star = stat_T_star['eta']
    eta_C = data['eta_C']

    kernel = skew_t_lpdf(y_grid[:, None],
                         data['nu'][None, ...],
                         data['mu'][None, ...],
                         data['sigma'][None, ...],
                         data['phi'][None, ...])

    lpdf_T = logsumexp(np.log(eta_T_star[None, ...]) + kernel, axis=-1)
    lpdf_C = logsumexp(np.log(eta_C[None, ...]) + kernel, axis=-1)

    return dict(pdf_T=np.exp(lpdf_T), pdf_C=np.exp(lpdf_C), y_grid=y_grid)

def plot_data(yT, yC, tcolor='red', ccolor='blue', bins=50, alpha=0.6,
              tlabel='T: Data', clabel='C: Data', 
              zero_T_pos=-10, zero_C_pos=-9, 
              t0label='T: prop. zeros', c0label='C: prop. zeros', 
              use_hist=False, lw=2, ls=":"):
    if use_hist:
        plt.hist(rm_inf(yT), color=tcolor, histtype='stepfilled', bins=bins,
                 alpha=alpha, density=True, label=tlabel)
        plt.hist(rm_inf(yC), color=ccolor, histtype='stepfilled', bins=bins,
                 alpha=alpha, density=True, label=clabel)
    else:
        sns.kdeplot(rm_inf(yT), color=tcolor, label=tlabel, ls=ls, lw=lw)
        sns.kdeplot(rm_inf(yC), color=ccolor, label=clabel, ls=ls, lw=lw)

    plt.scatter(zero_T_pos, np.isinf(yT).mean(),
                s=100, color=tcolor, alpha=alpha, label=t0label)
    plt.scatter(zero_C_pos, np.isinf(yC).mean(),
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
                eta_C=eta_C, eta_T=eta_T, mu=loc, sigma=scale, nu=nu, 
                phi=phi)

def prior_samples(B, stan_data):
    K = stan_data['K']
    p = np.random.beta(stan_data['a_p'], stan_data['b_p'], B)
    gamma_C  = np.random.beta(stan_data['a_gamma'], stan_data['b_gamma'], B)
    gamma_T  = np.random.beta(stan_data['a_gamma'], stan_data['b_gamma'], B)
    gamma_T_star = compute_gamma_T_star(gamma_C=gamma_C, gamma_T=gamma_T, p=p)
    eta_C  = np.random.dirichlet(stan_data['a_eta'], B)
    eta_T  = np.random.dirichlet(stan_data['a_eta'], B)
    eta_T_star = compute_eta_T_star(eta_C=eta_C, eta_T=eta_T, p=p[:, None])
    mu = np.random.normal(stan_data['mu_bar'], stan_data['s_mu'], (B, K))
    sigma = 1 / np.random.gamma(stan_data['a_sigma'], 1/stan_data['b_sigma'], (B, K))
    nu = np.random.lognormal(stan_data['m_nu'], stan_data['s_nu'], (B, K))
    phi = np.random.normal(stan_data['m_phi'], stan_data['s_phi'], (B, K))
    return dict(p=p, K=K,
                gamma_C=gamma_C, gamma_T=gamma_T, gamma_T_star=gamma_T_star,
                eta_C=eta_C, eta_T=eta_T, eta_T_star=eta_T_star,
                mu=mu, sigma=sigma, nu=nu, phi=phi)


if __name__ == '__main__':
    print('Testing...')

    data = gen_data(11, 9, p=.6, gamma_C=.4, gamma_T=.6, K=4)

    print('Tests execute successfully.')
    
    x = rand_skew_t(30, 2, .5, -3, size=10000)
    plt.hist(x, bins=100, density=True)
    xx = np.linspace(-1, 4, 1000)
    plt.plot(xx, skew_t_pdf(xx, 30, 2, .5, -3))
    plt.show()
