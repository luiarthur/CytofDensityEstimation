import numpy as np
from scipy.special import logit, expit


def default_state(data, priors):
    pass


class Model:
    def __init__(self, data, priors, state=None):
        if state is None:
            state = default_state(data, priors)
        self.data = data
        self.priors = priors
        self.state = state

    def update(self, priors):
        self.update_p(priors)
        self.update_beta(priors, data)

    def update_p(self, priors):
        a = priors.a_p + self.beta
        b = priors.b_p + 1 - self.beta
        self.p = np.random.beta(a, b)

    def update_beta(self, priors, data):
        logit_pnew = np.logit(p) + data.ZT * (np.log(self.gammaT) - np.log(self.gammaC))
        logit_pnew += data.NT_finite * (np.log1p(-self.gammaT) - np.log1p(-self.gammaC))


