import Pkg; Pkg.activate("../..")
using StatsPlots
import Random

Random.seed!(0)
X = cumsum(randn(10, 4) .+ 1, dims=1) * .1 .+ [1 2 3 4]
plot(X, legendfontsize=10, m=:square, ms=1, msc=:white, legend=:outerright,
     lw=5, palette=:tab10, msw=0)
