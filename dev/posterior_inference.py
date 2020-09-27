import numpy as np
from scipy.stats import truncnorm

def rand_skew_t(nu, loc, scale, alpha):
    w = np.random.gamma(nu/2, 2/nu)
    z = truncnorm.rvs(0, np.inf, scale=np.sqrt(1/w))
    delta = alpha / np.sqrt(1 + alpha**2)
    return loc + scale * z * delta + scale * np.sqrt(1 - delta**2) * np.random.randn()

def post_pred(fit, seed=None):
    if seed is not None:
        np.random.seed(seed)

    B = fit['p'].shape[0]
    return np.vstack([one_post_pred(fit, b) for b in range(B)])

def one_post_pred(fit, b):
    K = fit['eta_T'].shape[1]
    yT_is_zero = np.random.binomial(1, fit['gamma_T'][b])
    yT = None

    if yT_is_zero:
        yT = -np.inf
    else:
        k = np.random.choice(np.arange(K), size=1, p=fit['eta_T'][b])[0]
        yT = rand_skew_t(fit['nu'][b, k],
                         fit['xi'][b, k],
                         fit['sigma'][b, k],
                         fit['phi'][b, k])

    yC_is_zero = np.random.binomial(1, fit['gamma_C'][b])
    yC = None

    if yC_is_zero:
        yC = -np.inf
    else:
        k = np.random.choice(np.arange(K), size=1, p=fit['eta_C'][b])[0]
        yC = rand_skew_t(fit['nu'][b, k],
                         fit['xi'][b, k],
                         fit['sigma'][b, k],
                         fit['phi'][b, k])
    return [yC, yT]

