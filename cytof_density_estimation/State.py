import numpy as np

class State:
    def __init__(self, p, beta, gammaC, gammaT, etaC, etaT, lambdaC, lambdaT,
                 mu, nu, omega, psi, vC, vT, zetaC, zetaT):
        self.p = p
        self.beta = beta
        self.gammaC = gammaC
        self.gammaT = gammaT
        self.etaC = etaC
        self.etaT = etaT
        self.lambdaC = lambdaC
        self.lambdaT = lambdaT
        self.mu = mu
        self.nu = nu
        self.omega = omega
        self.psi = psi
        self.vC = vC
        self.vT = vT
        self.zetaC = zetaC
        self.zetaT = zetaT

    def update(self, priors):
        pass

    def update_p(self, priors):
        a = priors.a_p + self.beta
        b = priors.b_p + 1 - self.beta
        self.p = np.random.beta(a, b)
