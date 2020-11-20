# Adaptive Metropolis Random Walk
# https://github.com/TuringLang/AdvancedMH.jl/blob/master/src/mh-core.jl

# NOTE:
# - Is there a more computationally efficient way?
# - Can we use bijectors to transform continuous parameters to unconstrained space?

import Pkg; Pkg.activate(".")
include("amh.jl")
using StatsPlots
using MCMCChains

K = 32
meantrue = zeros(K)
covtrue = let
  M = randn(K,K)
  M*M'
end

m = DensityModel(x -> logpdf(MvNormal(meantrue, covtrue), x))
# p = RWMH(MvNormal(K, .1))
p = RWAM(K)
c = sample(m, p, Int(1e5), chain_type=Chains);
c = c[50000:10:end];

plot(vec(get(c, :lp)[1].data), label=nothing)
mean(c).nt[:mean]

X = c.value.data[:, 1:end-1, 1]
covtrue
cov(X)
scatter(covtrue, cov(X), label=nothing, color=:blue, alpha=.3)
corrplot(X[:, 1:3], grid=false)

# boxplot(X, label=nothing)
