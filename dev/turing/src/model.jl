import Pkg; Pkg.activate(joinpath(@__DIR__, "../"))
using Turing
using StatsFuns
Turing.setadbackend(:tracker)

include(joinpath(@__DIR__, "Helper.jl"))

collect_finite(y) = y[isfinite.(y)]
count_inf(y) = sum(y .== -Inf)

compute_gammaT_star(gammaC, gammaT, p) = p * gammaT + (1 - p) * gammaC
function compute_etaT_star(etaC, etaT, p, gammaC, gammaT, gammaT_star)
  numer = p * etaT * (1 - gammaT) + (1 - p) * etaC * (1 - gammaC)
  denom = 1 - gammaT_star
  return numer / denom
end
function compute_star(etaC, etaT, p, gammaC, gammaT)
  gammaT_star = compute_gammaT_star(gammaC, gammaT, p)
  etaT_star = compute_etaT_star(etaC, etaT, p, gammaC, gammaT, gammaT_star)
  return (gammaT_star, etaT_star)
end

function mixlpdf(df, loc, scale, skew, y, w) 
  lm = logsumexp(Helper.skewt_logpdf.(df', loc', scale', skew', y) .+
                 log.(w)', dims=2)
  return sum(lm)
end

@model cytofdens(yC_finite, yT_finite, nC, nT, zC, zT, K) = begin
  p ~ Beta(.5, .5)
  gammaC ~ Beta(1, 1)
  gammaT ~ Beta(1, 1)
  etaT ~ Dirichlet(K, 1/K)
  etaC ~ Dirichlet(K, 1/K)

  gammaT_star = compute_gammaT_star(gammaC, gammaT, p)
  etaT_star = compute_etaT_star(etaC, etaT, p, gammaC, gammaT, gammaT_star)
  
  mu ~ filldist(Normal(0, 3), K)
  sigma ~ filldist(LogNormal(0, 0.5), K)
  nu ~ filldist(LogNormal(3.5, 0.5), K)
  phi ~ filldist(Normal(0, 3), K)

  zC ~ Binomial(nC, gammaC)
  zT ~ Binomial(nT, gammaT_star)

  Turing.acclogp!(_varinfo, mixlpdf(nu, mu, sigma, phi, yC_finite, etaC))
  Turing.acclogp!(_varinfo, mixlpdf(nu, mu, sigma, phi, yT_finite, etaT_star))
  # yC_finite .~ UnivariateGMM(mu, sigma, Categorical(etaC))
  # yT_finite .~ UnivariateGMM(mu, sigma, Categorical(etaT_star))
end


# NUTS
yC = [randn(100) .+ 1; fill(-Inf, 50)]
yT = [randn(100); fill(-Inf, 100)]
@time nuts_chain = begin
    K = 5
    n_samples = 500
    nadapt = 500
    iterations = n_samples + nadapt
    target_accept_ratio = 0.8
    
    yC_finite = collect_finite(yC)
    yT_finite = collect_finite(yT)
    nC = length(yC)
    nT = length(yT)
    zC = count_inf(yC)
    zT = count_inf(yT)
    sample(cytofdens(yC_finite, yT_finite, nC, nT, zC, zT, K),
           NUTS(nadapt, target_accept_ratio, max_depth=6),
           iterations);
end;

