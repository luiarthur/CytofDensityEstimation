import numpy as np
from scipy.special import logsumexp
from simulate_data import rand_skew_t, skew_t_lpdf
import matplotlib.pyplot as plt

# TODO: Vectorize!
def post_predictive(fit, seed=None):
    """
    Get posterior predictive samples.
    """
    if seed is not None:
        np.random.seed(seed)

    B = fit['p'].shape[0]
    return np.vstack([one_post_pred(fit, b) for b in range(B)])

# FIXME: This is wrong!
def one_post_pred(fit, b):
    print('This is wrong! Do not use this!')
    K = fit['eta_T'].shape[1]
    yT_is_zero = np.random.binomial(1, fit['gamma_T'][b])
    yT = None

    if yT_is_zero:
        yT = -np.inf
    else:
        k = np.random.choice(np.arange(K), size=1, p=fit['eta_T'][b])[0]
        # This should actually be a mixture because 
        # sample T is a mixture of C as well.
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

def post_density(fit, y_grid):
    """
    Get density of posterior predicrive over a grid of y.
    """
    kernel = skew_t_lpdf(y_grid[:, None, None], 
                         fit["nu"][None, ...],
                         fit["xi"][None, ...],
                         fit["sigma"][None, ...],
                         fit["phi"][None, ...])
    eta_T_star = fit["pnot0_T"] / (1 - fit['p0_T'][:, None])
    lpdf_T = logsumexp(np.log(eta_T_star[None, ...]) + kernel, axis=-1)
    lpdf_C = logsumexp(np.log(fit["eta_C"][None, ...]) + kernel, axis=-1)
    return dict(pdf_T=np.exp(lpdf_T), pdf_C=np.exp(lpdf_C))


def plot_ci(x, loc, a=0.05, **kwargs):
    x_lower = np.quantile(x, a / 2)
    x_upper = np.quantile(x, 1 - a / 2)
    plt.plot([loc, loc], [x_lower, x_upper], **kwargs)


def plot_post_predictive_density(fit, y_grid, tlabel='T: post. pred.',
                                 clabel='C: post. pred.', return_grid=False,
                                 tcolor='red', ccolor='blue', fill_alpha=0.3,
                                 mean_alpha=1):
    post_dens = post_density(fit, y_grid)

    upper_T = np.percentile(post_dens['pdf_T'], 97.5, axis=-1)
    lower_T = np.percentile(post_dens['pdf_T'], 2.5, axis=-1)
    upper_C = np.percentile(post_dens['pdf_C'], 97.5, axis=-1)
    lower_C = np.percentile(post_dens['pdf_C'], 2.5, axis=-1)

    mean_T = np.mean(post_dens['pdf_T'], axis=-1)
    mean_C = np.mean(post_dens['pdf_C'], axis=-1)

    if fill_alpha > 0:
        _tlabel = '' if mean_alpha > 0 else tlabel
        _clabel = '' if mean_alpha > 0 else clabel
        plt.fill_between(y_grid, lower_T, upper_T, alpha=fill_alpha, color=tcolor,
                         label=_tlabel, lw=0)
        plt.fill_between(y_grid, lower_C, upper_C, alpha=fill_alpha, color=ccolor,
                         label=_clabel, lw=0)

    if mean_alpha > 0:
        plt.plot(y_grid, mean_C, alpha=mean_alpha, color=ccolor, label=clabel)
        plt.plot(y_grid, mean_T, alpha=mean_alpha, color=tcolor, label=tlabel)


def print_stat(param, fit, truth=None, digits=2):
    m = np.round(fit[param].mean(0), digits)
    s = np.round(fit[param].std(0), digits)

    msg = f'{param}: mean={m}' 

    if truth is not None:
        t = np.round(truth[param], digits)
        msg += f', truth={t}'

    msg += f', sd={s}'

    print(msg)

def print_summary(fit, truth=None, digits=2):
    print_stat('p', fit, truth=truth, digits=digits)
    print_stat('gamma_T', fit, truth=truth, digits=digits)
    print_stat('gamma_C', fit, truth=truth, digits=digits)
    print_stat('sigma', fit, truth=truth, digits=digits)
    print_stat('phi', fit, truth=truth, digits=digits)
    print_stat('xi', fit, truth=truth, digits=digits)
    print_stat('nu', fit, truth=truth, digits=digits)
    print_stat('eta_T', fit, truth=truth, digits=digits)
    print_stat('eta_C', fit, truth=truth, digits=digits)

