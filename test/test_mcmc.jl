#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
using CytofDensityEstimation; const CDE=CytofDensityEstimation
using Distributions

include("runtests.jl")
=#


@testset "metropolis adaptive" begin
  easy = true
  if easy 
    K = 3
    S = CDE.MCMC.eye(K) * 1.0
    S[1,2] = S[2,1] = -0.8
    S[1,3] = S[3,1] = 0.5
  else
    K = 32
    S = let
      M = randn(K,K)
      M*M'
    end
  end
  inv_S = inv(S)
  log_prob(x) = -x' * inv_S * x / 2.0
  mvn = MvNormal(S)

  tuner = CDE.MCMC.MvTuner(K)
  out = [randn(K)]
  for _ in 1:20000
    xnew = CDE.MCMC.metropolisAdaptive(last(out), log_prob, tuner)
    append!(out, [xnew])
  end
  cov(out[10000:end])[1:5, 1:5]
  S[1:5, 1:5]
  # plot([getindex.(out, 1) getindex.(out, 2)])
  # plot(getindex.(out, 1), getindex.(out, 4))
end
