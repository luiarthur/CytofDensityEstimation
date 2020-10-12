#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#
# ApproximateTwoSampleKSTest(a, b)

@testset "Test fit" begin
  K = 3
  # use_big_data = true
  use_big_data = false

  if use_big_data
    data = CDE.Model.Data([fill(-Inf, 30000); randn(70000)],
                          [fill(-Inf, 20000); randn(60000)])
  else
    # data = CDE.Model.Data([-Inf, 1.0, 2.0], [-1.0, -Inf, 3.0, 4.0])
    data = CDE.Model.Data([fill(-Inf, 300); randn(700)],
                          [fill(-Inf, 300); randn(700)])
  end

  prior = CDE.Model.Prior(K)
  state = CDE.Model.State(data, prior)
  tuners = CDE.Model.Tuners(K)

  chain, laststate, summarystats = CDE.Model.fit(state, data, prior, tuners)

  println(mean(CDE.Model.group(:p, chain)))
  println(mean(CDE.Model.group(:beta, chain)))
end
