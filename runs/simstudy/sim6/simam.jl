println("Compile libraries on main processor..."); flush(stdout)
include("imports.jl")
include("scenarios.jl")
println("Finished loading libraries."); flush(stdout)

simdata = scenarios(1, seed=1, Ni=10000)
data = CDE.Model.Data(simdata[:yC], simdata[:yT])
K = 6
prior = CDE.Model.PriorAM(K, true)
println(prior)

warmup() = let
  init = CDE.Model.StateAM(data, prior)
  tuner = CDE.Model.MCMC.MvTuner(CDE.Model.tovec(init))
  for n in 1:10
    @time chain, init, summarystats, tuner = CDE.amfit(
      init, data, prior, nsamps=[1], nburn=200, tuner=tuner, nc=50*n, nt=50*n)
    tuner.iter = 1
  end
  return init, tuner
end
init, tuner = warmup()

@time chain, laststate, summarystats, tuner = CDE.amfit(
  init, data, prior, thin=2, nsamps=[2000], nburn=2000,
  tuner=tuner)

function convert_chain!(chain, beta)
  for d in chain[1]
    d[:psi] = CDE.Util.toaltskew.(d[:sigma], d[:phi])
    d[:omega] = CDE.Util.toaltscale.(d[:sigma], d[:phi]) .^ 2
    d[:beta] = beta
  end
end
convert_chain!(chain, true)

imgdir = "results/testam/img"; mkpath(imgdir)
CDE.Model.plotpostsummary([chain[1][end-200:end]], summarystats, simdata[:yC],
                          simdata[:yT], imgdir; digits=3, laststate=laststate,
                          bw_postpred=0.2, simdata=simdata,
                          ygrid=default_ygrid(), xlims_=nothing,
                          plotsize=(400,400), density_legend_pos=:topleft)
 
