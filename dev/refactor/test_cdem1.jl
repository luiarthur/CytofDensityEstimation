import Pkg; Pkg.activate("../../")
using StatsPlots
include("cdem1.jl")

yC = randn(100)
yT = randn(100) .+ 1
K = 5
m, spl = generate_model_and_sampler(yC, yT, K, eps=0.01, L=50)

@time chain = sample(m, spl, 500, discard_initial=200)
