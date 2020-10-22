#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#
# ApproximateTwoSampleKSTest(a, b)

@testset "Test update!" begin
  K = 3
  # use_big_data = true
  use_big_data = false

  if use_big_data
    data = CDE.Model.Data([fill(-Inf, 30000); randn(70000)],
                          [fill(-Inf, 20000); randn(60000)])
  else
    data = CDE.Model.Data([-Inf, 1.0, 2.0], [-1.0, -Inf, 3.0, 4.0])
  end

  prior = CDE.Model.Prior(K)
  state = CDE.Model.State(data, prior)
  tuners = CDE.Model.Tuners(K)

  @time for i in ProgressBar(1:100)
    CDE.Model.update_state!(state, data, prior, tuners)
  end

  # Test flags
  flags = [CDE.UpdateBetaWithSkewT(), CDE.UpdateLambdaWithSkewT()]
  mu_orig = deepcopy(state.mu)
  nu_orig = deepcopy(state.nu)
  @time for i in ProgressBar(1:100)
    CDE.Model.update_state!(state, data, prior, tuners, fix=[:mu], flags=flags)
  end
  @test mu_orig == state.mu
  @test nu_orig != state.nu
end
