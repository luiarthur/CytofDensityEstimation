from scipy.stats import truncnorm, t
import numpy as np
import matplotlib.pyplot as plt

def rm_inf(x):
    return list(filter(lambda x: not np.isinf(x), x))

def rand_skew_t(nu, loc, scale, alpha, size=None):
    w = np.random.gamma(nu/2, 2/nu, size=size)
    z = truncnorm.rvs(0, np.inf, scale=np.sqrt(1/w), size=size)
    delta = alpha / np.sqrt(1 + alpha**2)
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


def plot_data(yT, yC, tcolor='red', ccolor='blue', bins=50, alpha=0.6,
              tlabel='T: Data', clabel='C: Data', zero_pos=-10,
              t0label='T: prop. zeros', c0label='C: prop. zeros'):
    plt.hist(rm_inf(yT), color=tcolor, histtype='stepfilled', bins=bins,
             alpha=alpha, density=True, label=tlabel)
    plt.hist(rm_inf(yC), color=ccolor, histtype='stepfilled', bins=bins,
             alpha=alpha, density=True, label=clabel)

    plt.scatter(zero_pos, np.isinf(yT).mean(),
                s=100, color=tcolor, alpha=alpha, label=t0label)
    plt.scatter(zero_pos, np.isinf(yC).mean(),
                s=100, color=ccolor, alpha=alpha, label=c0label)
 

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
    yT = sample(n_T, stat_T_star['gamma'], stat_T_star['eta'],
                loc, scale, nu, alpha)

    print('Tests execute successfully.')
    
    x = rand_skew_t(30, 2, .5, -3, size=10000)
    plt.hist(x, bins=100, density=True)
    xx = np.linspace(-1, 4, 1000)
    plt.plot(xx, skew_t_pdf(xx, 30, 2, .5, -3))
    plt.show()
