from scipy.special import expit, logit, logsumexp
from scipy import stats
import matplotlib.pyplot as plt
import pystan
import numpy as np 
import mcmc
from tqdm import trange

def update_beta(y, p, sd):
    log_numer = np.log(p) - sum((y - 1) ** 2) / (2 * sd * sd)
    log_denom = np.log1p(-p) - sum(y ** 2) / (2 * sd * sd)
    p1 = expit(log_numer - log_denom)
    return p1 > np.random.rand()

def update_p(y, beta):
    return np.random.beta(.5 + beta, .5 + 1 - beta)

def conjugate_run(y, nsamps=1000, nburn=1000, p=0.5, beta=1, sd=0.1):
    out = dict(p=[], beta=[])

    for i in trange(nburn + nsamps):
        beta = update_beta(y, p, sd)
        p = update_p(y, beta)
        if i >= nburn:
            out['p'] += [p]
            out['beta'] += [beta]

    return out

def mh_run(y, nsamps=1000, nburn=2000, p=0.5, sd=0.1, stepsize=0.1):
    tuner = mcmc.Tuner(stepsize)

    def log_prob(p):
        log_prior = stats.beta(.5, .5).logpdf(p)
        log_like_1 = np.log(p) + mcmc.lpdf_normal(y, 1, sd)
        log_like_0 = np.log1p(-p) + mcmc.lpdf_normal(y, 0, sd)
        log_like = np.stack([log_like_1, log_like_0], axis=-1)
        log_like = logsumexp(log_like, axis=-1).sum()
        return log_prior + log_like

    def update(p, tuner):
        return mcmc.ametropolis_unit_var(p, log_prob, tuner)
        return p

    out = np.zeros(nsamps)
    for i in trange(nsamps + nburn):
        p = update(p, tuner)
        if i >= nburn:
            out[i - nburn] = p
    
    return out


stan_model = """
data {
  int<lower=0> N;
  real y[N];
  real<lower=0> s;
}

parameters {
  real<lower=0, upper=1> p;
}

model {
  vector[N] ll;

  p ~ beta(0.5, 0.5);
  
  for (i in 1:N) {
    ll[i] = log_mix(p,
                    normal_lpdf(y[i] | 1, s),
                    normal_lpdf(y[i] | 0, s));
  }

  target += sum(ll);
}
"""

if __name__ == '__main__':
    mu = 1
    sd = 0.1
    y = np.random.randn(5) * sd + mu
    out_gibbs = conjugate_run(y, sd=sd)

    # plt.hist(y); plt.show()
    # plt.hist(out['p'], bins=50); plt.show()

    print("---------------- RESULTS --------------------")
    print(f'Gibbs:')
    print(f'beta: {np.mean(out_gibbs["beta"])}')
    print(f'p: {np.mean(out_gibbs["p"])}')

    # sm = pystan.StanModel(model_code=stan_model)
    # stan_data = dict(y=y, s=sd, N=len(y))
    # fit = sm.sampling(data=stan_data, iter=2000, warmup=1000, chains=1, seed=1)
    # nuts_beta_mean = np.mean([update_beta(y, p, sd) for p in fit['p']])
    # print(f'NUTS:')
    # print(f'beta: {nuts_beta_mean}')
    # print(f'p: {np.mean(fit["p"])}')

    out_mh = mh_run(y, sd=sd)
    print("---------------- RESULTS --------------------")
    print(f'MH:')
    # print(f'beta: {np.mean(out_mh["beta"])}')
    print(f'p: {np.mean(out_mh)}')
    plt.hist(out_mh, bins=30, label='MH', color='blue', alpha=0.3, histtype='stepfilled');
    plt.hist(out_gibbs['p'], bins=30, label='Gibbs', color='red', alpha=0.3, histtype='stepfilled');
    plt.legend()
    plt.show()

    # Question: Why are the results different (for p ) after marginalizing 
    # over beta?
