import numpy as np
from scipy.special import logsumexp
from simulate_data import rand_skew_t, skew_t_lpdf
import matplotlib.pyplot as plt

def prior_predictive_observed_samples(n, seed=None):
    NotImplemented
    # TODO:
    # Create samps dictionary for parameters.
    # Args: prior or distributions.
    return predictive_observed_samples(samps, seed=seed)

def post_predictive_observed_samples(fit, seed=None):
    return predictive_observed_samples(fit, seed=seed)

def predictive_observed_samples(samps, seed=None):
    if seed is not None:
        np.random.seed(seed)

    B, K = samps['nu'].shape

    comp_C = np.stack([
        np.random.choice(K, size=1, replace=True, p=eta_C)
        for eta_C in samps['eta_C']])[:, 0]

    comp_T = np.stack([
        np.random.choice(K, size=1, replace=True, p=eta_T)
        for eta_T in samps['eta_T_star']])[:, 0]

    y_C = rand_skew_t(np.choose(comp_C, samps['nu'].T),
                      np.choose(comp_C, samps['mu'].T),
                      np.choose(comp_C, samps['sigma'].T),
                      np.choose(comp_C, samps['phi'].T))

    y_T = rand_skew_t(np.choose(comp_T, samps['nu'].T),
                      np.choose(comp_T, samps['mu'].T),
                      np.choose(comp_T, samps['sigma'].T),
                      np.choose(comp_T, samps['phi'].T))

    return dict(y_C=y_C, y_T=y_T)


def post_density(fit, y_grid, use_eta_T=False):
    """
    Get density of posterior predicrive over a grid of y.
    """
    kernel = skew_t_lpdf(y_grid[:, None, None], 
                         fit["nu"][None, ...],
                         fit["mu"][None, ...],
                         fit["sigma"][None, ...],
                         fit["phi"][None, ...])
    lpdf_C = logsumexp(np.log(fit["eta_C"][None, ...]) + kernel, axis=-1)

    if use_eta_T:
        lpdf_T = logsumexp(np.log(fit['eta_T'][None, ...]) + kernel, axis=-1)
    else:
        lpdf_T = logsumexp(np.log(fit['eta_T_star'][None, ...]) + kernel, axis=-1)

    return dict(pdf_T=np.exp(lpdf_T), pdf_C=np.exp(lpdf_C))


def plot_ci(x, loc, a=0.05, **kwargs):
    x_lower = np.quantile(x, a / 2)
    x_upper = np.quantile(x, 1 - a / 2)
    plt.plot([loc, loc], [x_lower, x_upper], **kwargs)


def plot_post_density(fit, y_grid, tlabel='T: post. pred.',
                      clabel='C: post. pred.', return_grid=False,
                      tcolor='red', ccolor='blue', fill_alpha=0.3,
                      mean_alpha=1, digits=3, a=0.05, use_eta_T=False):
    post_dens = post_density(fit, y_grid, use_eta_T=use_eta_T)

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

    p_lower = np.round(np.quantile(fit['p'], a / 2), digits)
    p_upper = np.round(np.quantile(fit['p'], 1 - a / 2), digits)
    p_mean = np.round(np.mean(fit['p']), digits)
    title = r'Prob($F_C \ne F_T$ | data)$\approx$' + f"{p_mean} {(p_lower, p_upper)}"
    plt.title(title)


def print_stat(param, fit, truth=None, digits=3, show_sd=True):
    m = np.round(fit[param].mean(0), digits)
    s = np.round(fit[param].std(0), digits)

    msg = f'{param}: mean={m}' 

    if truth is not None:
        t = np.round(truth[param], digits)
        msg += f', truth={t}'

    if show_sd:
        msg += f', sd={s}'

    print(msg)

def print_summary(fit, truth=None, digits=3, show_sd=True):
    print_stat('p', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('gamma_C', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('gamma_T_star', fit, digits=digits, show_sd=show_sd)
    print_stat('gamma_T', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('sigma', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('phi', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('mu', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('nu', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('eta_C', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('eta_T_star', fit, digits=digits, show_sd=show_sd)
    print_stat('eta_T', fit, truth=truth, digits=digits, show_sd=show_sd)

