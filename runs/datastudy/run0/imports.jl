ENV["GKSwstype"] = "nul"  # For StatsPlots

import Pkg; Pkg.activate(joinpath(@__DIR__, "../../../"))
using CSV, DataFrames
using CytofDensityEstimation
const CDE = CytofDensityEstimation
using Distributions
import Random
using BSON
using StatsPlots

using RCall
@rimport stats as rstats 


function postprocess(chain, laststate, summarystats, yC, yT, imgdir;
                     bw_postpred=0.2, density_legend_pos=:best,
                     ygrid=collect(range(-8, 8, length=1000)))
  # Print KS Statistic.
  ks_fit = rstats.ks_test(yC, yT)
  println("KS-test p-value: ", ks_fit["p.value"][1])
  flush(stdout)

  # Print summary statistics.
  CDE.Model.printsummary(chain, summarystats)
  flush(stdout)

  # Plot results.
  @time CDE.Model.plotpostsummary(chain, summarystats, yC, yT, imgdir,
                                  bw_postpred=bw_postpred, ygrid=ygrid,
                                  density_legend_pos=density_legend_pos)
  flush(stdout)
end


function defaults(yC, yT, K; seed=nothing)
  seed == nothing || Random.seed!(seed)

  data = CDE.Model.Data(yC, yT)
  prior = CDE.Model.Prior(K, mu=CDE.Model.compute_prior_mu(data),
                          nu=LogNormal(1.6, 0.4), p=Beta(100, 100),
                          omega=InverseGamma(.1, .1), psi=Normal(-1, 1))
  state = CDE.Model.State(data, prior)
  tuners = CDE.Model.Tuners(K, 0.1)
  return (state=state, data=data, prior=prior, tuners=tuners)
end


function run(config)
  resultsdir = config[:resultsdir]
  mkpath(resultsdir)
  path_to_log = joinpath(resultsdir, "log.txt")
  CDE.Util.redirect_stdout_to_file(path_to_log) do
    _run(config)
  end
end

function _run(config)
  println("pid: $(getpid())"); flush(stdout)

  resultsdir = config[:resultsdir]
  awsbucket = config[:awsbucket]
  yC = config[:yC]
  yT = config[:yT]
  K = config[:K]
  nsamps = config[:nsamps]
  nburn = config[:nburn]

  # Make image path id needed.
  imgdir = joinpath(resultsdir, "img")
  mkpath(imgdir)

  # Create initial state, data, prior, and tuners objects.
  state, data, prior, tuners = defaults(yC, yT, K, seed=0)
  println("Priors:")
  foreach(fn -> println(fn, " => ", getfield(prior, fn)),
          fieldnames(CDE.Model.Prior))
  
  # Run analysis.
  println("Run Chain ..."); flush(stdout)
  @time chain, laststate, summarystats = CDE.Model.fit(
      state, data, prior, tuners, nsamps=[nsamps], nburn=nburn, thin=1,
      rep_aux=10, rep_beta_flipped=50)
  flush(stdout)

  # Save results
  BSON.bson("$(resultsdir)/results.bson",
            Dict(:chain=>chain, :laststate=>laststate, :data=>data,
                 :prior=>prior, :summarystats=>summarystats))

  # Load data:
  out = BSON.load("$(resultsdir)/results.bson")

  # Post process
  postprocess(out[:chain], out[:laststate], out[:summarystats], out[:data].yC,
              out[:data].yT, imgdir; bw_postpred=0.2,
              ygrid=collect(range(-8, 8, length=1000)),
              density_legend_pos=:topleft)

  # Send results
  if awsbucket != nothing
    CDE.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
  end

  # Print done.
  println("Done!"); flush(stdout)
end
