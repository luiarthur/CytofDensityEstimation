#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#

ENV["GKSwstype"] = "nul"  # For StatsPlots

@testset "Test fit" begin
  K = 3
  # use_big_data = true
  use_big_data = false

  if use_big_data
    data = CDE.Model.Data([fill(-Inf, 30000); randn(70000)],
                          [fill(-Inf, 20000); randn(60000)])
  else
    # data = CDE.Model.Data([-Inf, 1.0, 2.0], [-1.0, -Inf, 3.0, 4.0])
    data = CDE.Model.Data([fill(-Inf, 300); randn(700) .+ 1],
                          [fill(-Inf, 300); randn(700) .+ 1])
    # NOTE: When the two distributions are the same, E[beta|data,param]
    # approaches 50% because the parameters gammaC, gammaT, etaC, etaT can be
    # qualitatively similar. So, p needs to be informatively small apriori?
  end

  prior = CDE.Model.Prior(K, p=Beta(1, 99))
  println(prior)
  state = CDE.Model.State(data, prior)
  tuners = CDE.Model.Tuners(K)

  chain, laststate, summarystats = CDE.Model.fit(state, data, prior, tuners,
                                                 nsamps=[500], nburn=0)

  CDE.Model.printsummary(chain, summarystats)

  imgdir = "img/test"; mkpath(imgdir)
  @time CDE.Model.plotpostsummary(chain, summarystats, data.yC, data.yT,
                                  imgdir, bw_postpred=0.2,
                                  ygrid=CDE.Model.default_ygrid())
end
