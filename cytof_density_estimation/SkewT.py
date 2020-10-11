import numpy as np
from scipy.stats import truncnorm, t
from scipy.special import logsumexp


def rand_skew_t(df, loc, scale, skew, size=None, delta=None):
    w = np.random.gamma(df/2, 2/df, size=size)
    inv_sqrt_w = np.sqrt(1 / w)
    z = truncnorm.rvs(0, np.inf, scale=inv_sqrt_w, size=size)
    if delta is None:
        delta = skew / np.sqrt(1 + skew**2)
    return (loc + scale * z * delta + 
            scale * np.sqrt(1 - delta**2) * 
            np.random.normal(scale=inv_sqrt_w, size=size))


def skew_t_lpdf(x, df, loc, scale, skew):
    z = (x - loc) / scale
    u = skew * z * np.sqrt((df + 1) / (df + z * z));
    kernel = (t.logpdf(z, df, 0, 1) + 
              t.logcdf(u, df + 1, 0, 1))
    return kernel + np.log(2/scale)


class SkewT:
    def __init__(self, loc, scale, df, skew):
        self.loc = loc
        self.scale = scale
        self.df = df
        self.skew = skew
        self.delta = skew / np.sqrt(1 + skew ** 2)

    def to_alt_param(self):
        """Returns alternate (skew, scale)."""
        alt_skew = self.scale * self.delta
        alt_scale = self.scale * np.sqrt(1 - self.delta ** 2)
        return (alt_skew, alt_scale)

    def from_alt_param(self, alt_skew, alt_scale):
        """Returns original (skew, scale) given alternate skew and scale."""
        skew = alt_skew / alt_scale
        scale = np.sqrt(alt_skew ** 2 + alt_scale ** 2)
        return (skew, scale)
                    
    def sample(self, size):
        return rand_skew_t(df=self.df, loc=self.loc, scale=self.scale, 
                           skew=self.skew, size=size, delta=self.delta)

    def logpdf(self, x):
        return skew_t_lpdf(x, df=self.df, loc=self.loc, scale=self.scale,
                           skew=self.skew)

    def pdf(self, x):
        return np.exp(self.logpdf(x))


# FIXME
if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    print('Testing...')

    st = SkewT(df=np.random.lognormal(3.5, .5), loc=np.random.randn()*3,
               scale=np.random.rand(), skew=np.random.randn()*3)
    x = st.sample(int(1e6))
    plt.hist(x, bins=100 if x.shape[0] >= 10000 else None, density=True)
    sns.kdeplot(x, label='kde', lw=2)
    xx = np.linspace(-1, 4, 1000)
    plt.plot(xx, st.pdf(xx), label='pdf', lw=2)
    plt.legend()
    plt.show()
