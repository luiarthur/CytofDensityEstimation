#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#

@testset "Test amfit" begin
  K = 3
  # use_big_data = true
  use_big_data = false

  if use_big_data
    data = CDE.Model.Data([fill(-Inf, 30000); randn(70000)],
                          [fill(-Inf, 20000); randn(60000)])
  else
    # data = CDE.Model.Data([-Inf, 1.0, 2.0], [-1.0, -Inf, 3.0, 4.0])
    simdata = CDE.Model.generate_samples(NC=2000, NT=2000,
                                         gammaC=.2, gammaT=.2,
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
  init = CDE.Model.StateAM(data, prior)
  tuner = CDE.Model.MCMC.MvTuner(CDE.Model.tovec(init))
  # for _ in 1:10
  for _ in 1:1
    @time chain, laststate, summarystats, tuner = CDE.amfit(
      init, data, prior, nsamps=[1], nburn=1000, tuner=tuner, nc=100, nt=100)
    tuner.iter = 1
  end

  thin = 10
  @time chain, laststate, summarystats, tuner = CDE.amfit(
    # laststate, data, prior, thin=thin, nsamps=[2000], nburn=10000,
    laststate, data, prior, thin=thin, nsamps=[20], nburn=10,
    tuner=tuner)

  B = length(chain[1])
  ll = getindex.(summarystats, :loglike)[end-(thin*B):thin:end]
  plot(ll, label=nothing)
  savefig("img/llamfit.pdf")
  closeall()

  for sym in fieldnames(typeof(laststate))
    println("$(sym): ", mean(getindex.(chain[1], sym)))
  end
  gammaC = getindex.(chain[1], :gammaC)
  gammaT = getindex.(chain[1], :gammaT)

  accept = gammaC[2:end] .!== gammaC[1:end-1]
  mean_accept = cumsum(accept) ./ eachindex(accept)
  plot(mean_accept, label=nothing)
  savefig("img/accept.pdf")
  closeall()

  plot(gammaC, gammaT, label=nothing); savefig("img/gamma-trace.pdf"); closeall()
end
