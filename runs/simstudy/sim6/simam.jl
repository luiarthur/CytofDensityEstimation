println("Compile libraries on main processor..."); flush(stdout)
include("imports.jl")
include("scenarios.jl")
println("Finished loading libraries."); flush(stdout)

simdata = scenarios(1, seed=1, Ni=5000)
data = CDE.Model.Data(simdata[:yC], simdata[:yT])
K = 6
prior = CDE.Model.PriorAM(K, true, phi=Normal(-10, 5), data=data)
println(prior)

warmup(N) = let
  init = CDE.Model.StateAM(data, prior)
  tuner = CDE.Model.MCMC.MvTuner(CDE.Model.tovec(init))
  for n in 1:N
    println("n: $n")
    @time chain, init, summarystats, tuner = CDE.amfit(
      init, data, prior, nsamps=[1], nburn=200, tuner=tuner,
      nc=50*n, nt=50*n, temper=10)
    tuner.iter = 1
  end
  return init, tuner
end
init, tuner = warmup(0)

@time chain, laststate, summarystats, tuner = CDE.amfit(
  init, data, prior, thin=2, nsamps=[2000], nburn=5000,
  tuner=tuner, nc=500, nt=500, temper=1)

function convert_chain!(chain, beta)
  for d in chain[1]
    d[:psi] = CDE.Util.toaltskew.(d[:sigma], d[:phi])
    d[:omega] = CDE.Util.toaltscale.(d[:sigma], d[:phi]) .^ 2
    d[:beta] = beta
  end
end
convert_chain!(chain, true)

imgdir = "results/testam/img"; mkpath(imgdir)
CDE.Model.plotpostsummary([chain[1]], summarystats, simdata[:yC],
                          simdata[:yT], imgdir; digits=3, laststate=laststate,
                          bw_postpred=0.2, ygrid=ygrid, plotsize=(400,400),
                          density_legend_pos=:topleft, simdata=simdata)
