#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#
# TODO: ApproximateTwoSampleKSTest(a, b)

@testset "Test fit" begin
  K = 3
  # use_big_data = true
  use_big_data = false

  if use_big_data
    data = CDE.Model.Data([fill(-Inf, 30000); randn(70000)],
                          [fill(-Inf, 20000); randn(60000)])
  else
    # data = CDE.Model.Data([-Inf, 1.0, 2.0], [-1.0, -Inf, 3.0, 4.0])
    data = CDE.Model.Data([fill(-Inf, 300); randn(900)],
                          [fill(-Inf, 300); randn(700)])
    # NOTE: When the two distributions are the same, E[beta|data,param]
    # approaches 50% because the parameters gammaC, gammaT, etaC, etaT
    # can be qualitatively similar.
  end

  prior = CDE.Model.Prior(K, p=Beta(10, 90))
  println(prior)
  state = CDE.Model.State(data, prior)
  tuners = CDE.Model.Tuners(K)

  chain, laststate, summarystats = CDE.Model.fit(state, data, prior, tuners,
                                                 nsamps=[1000], nburn=0)

  p = CDE.Model.group(:p, chain)
  beta = CDE.Model.group(:beta, chain)
  gammaC = CDE.Model.group(:gammaC, chain)
  gammaT = CDE.Model.group(:gammaT, chain)
  nu = permutedims(reduce(hcat, CDE.Model.group(:nu, chain)), (2, 1))

  println("mean p: ", mean(p))
  println("mean beta: ", mean(beta))
  println("mean etaC: ", mean(CDE.Model.group(:etaC, chain)))
  println("mean etaT ", mean(CDE.Model.group(:etaT, chain)))
  println("mean gammaC: ", mean(gammaC))
  println("mean gammaT: ", mean(gammaT))
  println("mean nu ", mean(CDE.Model.group(:nu, chain)))

  skew, scale = CDE.Model.fetch_skewt_stats(chain)
  println("Mean skew: ", mean(skew))
  println("Mean scale: ", mean(scale))

  # Plots
  ll = [s[:loglike] for s in summarystats]
  plot(ll); savefig("img/ll.pdf"); closeall();
  plot(p); savefig("img/p.pdf"); closeall();
  plot(beta); savefig("img/beta.pdf"); closeall();
  plot(gammaC); savefig("img/gammaC.pdf"); closeall();
  plot(gammaT); savefig("img/gammaT.pdf"); closeall();
  plot(nu); savefig("img/nu.pdf"); closeall();
end
