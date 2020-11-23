# Adaptive Metropolis Random Walk
# https://github.com/TuringLang/AdvancedMH.jl/blob/master/src/mh-core.jl

# NOTE:
# - Is there a more computationally efficient way?
# - Can we use bijectors to transform continuous parameters to unconstrained space?

import Pkg; Pkg.activate(".")
include("amh.jl")
using StatsPlots
using MCMCChains
import Random

Random.seed!(0)

trilindex(x) = x[tril(ones(Int, size(x))) .== 1]
Base.range(x::AbstractArray) = [minimum(x), maximum(x)]
# x = reshape(collect(1:9), 3, 3)
# trilindex(x)

K = 6
meantrue = randn(K) # zeros(K)
covtrue = let
  M = randn(K,K)
  M*M'
end

m = DensityModel(x -> logpdf(MvNormal(meantrue, covtrue), x))
# p = RWMH(MvNormal(K, .1))
p = RWAM(K)
c = sample(m, p, Int(60000), chain_type=Chains);
c = c[30001:10:end];

plot(vec(get(c, :lp)[1].data), label=nothing)
meanx = mean(c).nt[:mean]

X = c.value.data[:, 1:end-1, 1]
covtrue
cov(X)
scatter(trilindex(covtrue), trilindex(cov(X)), label=nothing, color=:blue, alpha=.6)
plot!(range(covtrue), range(covtrue), ls=:dot, label=nothing)
corrplot(X[:, 1:3], grid=false)

scatter(meantrue, meanx, label=nothing, color=:blue, alpha=.6)
plot!(range(meantrue), range(meantrue), ls=:dot, label=nothing)

# boxplot(X, label=nothing)
