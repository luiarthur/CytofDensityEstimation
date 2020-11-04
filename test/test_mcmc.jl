#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
using CytofDensityEstimation; const CDE=CytofDensityEstimation
using Distributions

include("runtests.jl")
=#


@testset "metropolis adaptive" begin
  easy = false
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
  xnew = randn(K)
  out = Vector{Float64}[]
  @time for i in 1:100000
    xnew = CDE.MCMC.metropolisAdaptive(xnew, log_prob, tuner)
    i > 50000 && append!(out, [xnew])
  end
  println("$((cov(out)[1], S[1]))")
  println("$((cov(out)[2], S[2]))")
  # plot([getindex.(out, 1) getindex.(out, 2)])
  # plot(getindex.(out, 1), getindex.(out, 4))
end
