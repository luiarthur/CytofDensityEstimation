#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#

@testset "Test update!" begin
  K = 3
  data = CDE.Model.Data([-Inf, 1.0, 2.0], [-1.0, -Inf, 3.0, 4.0])
  prior = CDE.Model.Prior(K)
  state = CDE.Model.State(data, prior)
  tuners = CDE.Model.Tuners(K)

  CDE.Model.update_state!(state, data, prior, tuners)
end


