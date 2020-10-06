using Turing

collect_finite(y) = y[isfinite.(y)]
count_zeros(y) = sum(y .== 0)

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

mixlpdf

@model cytofdens(yC, yT, K)= begin
  nC = length(yC)
  nT = length(yT)
  zC = count_zeros(yC)
  zT = count_zeros(yT)
  yC_finite = collect_finite(yC)
  yT_finite = collect_finite(yT)

  p ~ Beta(.5, .5)
  gammaC ~ Beta(1, 1)
  gammaT ~ Beta(1, 1)
  etaT ~ Dirichlet(K, 1/K)
  etaC ~ Dirichlet(K, 1/K)

  gammaT_star, etaT_star = compute_star(etaC, etaT, p, gammaC, gammaT)
  
  mu ~ filldist(Normal(0, 3), K)
  sigma ~ filldist(LogNormal(0, 0.5), K)
  nu ~ filldist(LogNormal(3.5, 0.5), K)
  phi ~ filldist(Normal(0, 3), K)

  zC ~ Binomial(nC, gammaC)
  zT ~ Binomial(nT, gammaT_star)



end
