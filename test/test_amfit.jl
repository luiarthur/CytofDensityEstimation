#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#

# ENV["GKSwstype"] = "nul"  # For StatsPlots

@testset "Test amfit" begin
  K = 3
  # use_big_data = true
  use_big_data = false

  if use_big_data
    data = CDE.Model.Data([fill(-Inf, 30000); randn(70000)],
                          [fill(-Inf, 20000); randn(60000)])
  else
    # data = CDE.Model.Data([-Inf, 1.0, 2.0], [-1.0, -Inf, 3.0, 4.0])
    simdata = CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=.2, gammaT=.2,
                                         etaC=[.2, .8], etaT=[.8,.2],
                                         loc=[-1, 1], scale=[.3, .3],
                                         df=[10, 10], skew=[-10, 0])
    data = CDE.Model.Data(simdata[:yC], simdata[:yT])
    # NOTE: When the two distributions are the same, E[beta|data,param]
    # approaches 50% because the parameters gammaC, gammaT, etaC, etaT can be
    # qualitatively similar. So, p needs to be informatively small apriori?
  end

  prior = CDE.Model.PriorAM(K, p=0.5)
  println(prior)
  state = CDE.Model.StateAM(data, prior)
  tuner = CDE.Model.MCMC.MvTuner(6K)

  @time chain, laststate, summarystats, tuner = CDE.amfit(state, data, prior, tuner=tuner,
                                                          thin=2, nsamps=[50], nburn=100,
                                                          temper=100)

  plot(getindex.(summarystats, :loglike), label=nothing)
  savefig("img/llamfit.pdf")
  closeall()

  getindex.(chain[1], :mu)
  getindex.(chain[1], :sigma)
  getindex.(chain[1], :nu)
  getindex.(chain[1], :phi)
  getindex.(chain[1], :etaC)
  getindex.(chain[1], :etaT)
  getindex.(chain[1], :gammaC)
  getindex.(chain[1], :gammaT)
 end
