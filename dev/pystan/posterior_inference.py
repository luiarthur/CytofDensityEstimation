import numpy as np
from scipy.special import logsumexp, expit, logit  # expit = sigmoid = logistic
from simulate_data import rand_skew_t, skew_t_lpdf, prior_samples
import matplotlib.pyplot as plt

def prior_predictive_observed_samples(n, samps, seed=None):
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

def plot_prior_density(stan_data, y_grid, B, title=None, a=0.05, digits=3,
                       **kwargs):

    samples = prior_samples(B, stan_data)

    if title is None:
        # p_lower = np.round(np.quantile(samples['p'], a / 2), digits)
        # p_upper = np.round(np.quantile(samples['p'], 1 - a / 2), digits)
        # p_mean = np.round(np.mean(samples['p']), digits)
        # title = r'Prob($F_C \ne F_T$)$\approx$' + f"{p_mean} {(p_lower, p_upper)}"
        beta_mean = np.random.binomial(1, samples['p']).mean()
        title = r'Prob($F_C \ne F_T$)$\approx$' + f"{beta_mean}"

    plot_post_density(samples, y_grid,
                      tlabel='T: prior pred.', clabel='C: prior pred.',
                      title=title, **kwargs)

def plot_post_density(fit, y_grid, tlabel='T: post. pred.',
                      clabel='C: post. pred.', return_grid=False,
                      tcolor='red', ccolor='blue', fill_alpha=0.3,
                      mean_alpha=1, digits=3, a=0.05, use_eta_T=False,
                      zero_C_pos=-9, zero_T_pos=-10, gamma_alpha=0.5,
                      plot_gamma=True, title=None):
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

    if title is None:
        # p_lower = np.round(np.quantile(fit['p'], a / 2), digits)
        # p_upper = np.round(np.quantile(fit['p'], 1 - a / 2), digits)
        # p_mean = np.round(np.mean(fit['p']), digits)
        # title = r'Prob($F_C \ne F_T$ | data)$\approx$' + f"{p_mean} {(p_lower, p_upper)}"
        if 'beta' in fit:
            beta_mean = np.round(np.mean(fit['beta']), digits)
            title = r'Prob($F_C \ne F_T$ | data)$\approx$' + f"{beta_mean}"

    plt.title(title)

    if plot_gamma:
        plot_ci(fit['gamma_T_star'], loc=zero_T_pos, alpha=gamma_alpha,
                color=tcolor, lw=2)
        plot_ci(fit['gamma_C'], loc=zero_C_pos, alpha=gamma_alpha,
                color=ccolor, lw=2)

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

def print_summary(fit, truth=None, digits=3, show_sd=True,
                  include_gamma_eta_T=False):
    print_stat('p', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('gamma_C', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('gamma_T_star', fit, digits=digits, show_sd=show_sd)
    if include_gamma_eta_T:
        print_stat('gamma_T', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('eta_C', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('eta_T_star', fit, digits=digits, show_sd=show_sd)
    if include_gamma_eta_T:
        print_stat('eta_T', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('nu', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('mu', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('sigma', fit, truth=truth, digits=digits, show_sd=show_sd)
    print_stat('phi', fit, truth=truth, digits=digits, show_sd=show_sd)

def isfinite(x):
    return np.isinf(x) == False

def beta_posterior(fit, data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    B, K = fit['mu'].shape

    y_T = data['y_T']

    nT = y_T.shape[0]

    # number of zeros
    ZT = np.isinf(y_T).sum()

    # Finite y.
    yfinite_T = y_T[isfinite(y_T)]

    p = fit['p']
    gamma_T= fit['gamma_T']
    gamma_C = fit['gamma_C']
    eta_T= fit['eta_T']
    eta_C = fit['eta_C']

    kernel = skew_t_lpdf(yfinite_T[:, None, None],
                         nu=fit['nu'][None, ...],
                         loc=fit['mu'][None, ...],
                         scale=fit['sigma'][None, ...],
                         skew=fit['phi'][None, ...])

    kernel_numer = logsumexp(np.log(eta_T[None, ...]) +
                             kernel, axis=-1).sum(0)
    kernel_denom = logsumexp(np.log(eta_C[None, ...]) +
                             kernel, axis=-1).sum(0)

    logit = ZT * (np.log(gamma_T) - np.log(gamma_C))
    logit += (nT - ZT) * (np.log1p(-gamma_T) - np.log1p(-gamma_C))
    logit += kernel_numer + np.log(p) - (kernel_denom + np.log1p(-p))

    p1 = expit(logit)

    return p1 > np.random.rand(B)
