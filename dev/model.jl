import Pkg; Pkg.activate("../")

using StatsPlots
using Turing
using Distributions
using StatsFuns
import Random

function marginal_loglike(mu, sigma, etaC, etaT, gammaC, gammaT,
                          p, yC_finite, yT_finite, ZC, ZT)
  NC = length(yC_finite)
  NT = length(yT_finite)

  a = sum(logpdf.(UnivariateGMM(mu, sigma, Categorical(etaC)), yC_finite))
  b = sum(logpdf.(UnivariateGMM(mu, sigma, Categorical(etaT)), yT_finite))
  c = sum(logpdf.(UnivariateGMM(mu, sigma, Categorical(etaC)), yT_finite))

  d = ZC * log(gammaC) + (NC - ZC) * log1p(-gammaC)
  e = ZT * log(gammaT) + (NT - ZT) * log1p(-gammaT)
  f = ZT * log(gammaC) + (NT - ZT) * log1p(-gammaC)

  return (a + d) + logsumexp([log(p) + b + e, log1p(-p) + c + f])
end

@model GMM_mixture(yC_finite, yT_finite, ZC, ZT, K, p) = begin
    mu ~ filldist(Normal(0, 1), K)
    sigmasq ~ filldist(InverseGamma(3, 2), K)
    sigma = sqrt.(sigmasq)

    etaC ~ Dirichlet(K, 1/K)
    etaT ~ Dirichlet(K, 1/K)

    gammaC ~ Beta(1, 1)
    gammaT ~ Beta(1, 1)

    log_target = marginal_loglike(mu, sigma, etaC, etaT, gammaC, gammaT,
                                  p, yC_finite, yT_finite, ZC, ZT)

    Turing.acclogp!(_varinfo, log_target)

    return (mu=mu, sigma=sigma, eta=eta)
end;

# SLOW.
Random.seed!(1);
m = GMM_mixture(randn(900), randn(900) .+ 1, 100, 100, 5, 0.5)
@time chain = sample(m, NUTS(500, 0.65), 1000)
