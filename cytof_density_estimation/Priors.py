class Priors:
    def __init__(self, K, a_p=0.5, b_p=0.5, a_gamma=1, b_gamma=1,
                 a_eta=None, m_mu=0, s_mu=3,
                 m_nu=3.5, s_nu=0.5, a_omega=3, b_omega=2, m_psi=-1, s_psi=3):

        if a_eta is None:
            a_eta = np.ones(K) / K

        self.a_p = a_p
        self.b_p = b_p
        self.a_gamma = a_gamma
        self.b_gamma = b_gamma
        self.a_eta = a_eta
        self.m_mu = m_mu
        self.s_mu = s_mu
        self.m_nu = m_nu
        self.s_nu = s_nu
        self.a_omega = a_omega
        self.b_omega = b_omega
        self.m_psi = m_psi
        self.s_psi = s_psi
