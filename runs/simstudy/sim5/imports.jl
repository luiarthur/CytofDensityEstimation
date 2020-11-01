ENV["GKSwstype"] = "nul"  # For StatsPlots

import Pkg; Pkg.activate(joinpath(@__DIR__, "../../../"))
using CytofDensityEstimation
const CDE = CytofDensityEstimation
using LaTeXStrings
using Distributions
using StatsPlots
import Random
using RCall
using BSON
import CytofDensityEstimation.Model.default_ygrid
using StatsFuns
@rimport stats as rstats 
VD = Vector{Dict{Symbol, Any}}

ygrid = collect(range(-10, 8, length=200))
default_plotsize = (400, 400)

# Plot true data density.
function plot_true_data_density(simdata, imgdir; lw=3,
                                plotsize=default_plotsize,
                                ygrid=ygrid, legendpos=:topleft)
  plot(ygrid, pdf.(simdata[:mmC], ygrid), lw=lw,
       label=L"y_C", color=:blue)
  plot!(ygrid, pdf.(simdata[:mmT], ygrid), lw=lw,
        label=L"y_T", color=:red)
  plot!(size=plotsize, legend=:topleft)
  p0C, p0T = simdata[:gammaC], simdata[:gammaT]
  title!("prop. -∞ in C: $(p0C) | prop. -∞ in T: $(p0T)", titlefont=font(10))
  savefig(joinpath(imgdir, "data-true-density.pdf"))
  closeall()
end

# Plot observed data histogram.
function plot_observed_hist(yC, yT, imgdir; bins, binsT=nothing,
                            alpha=0.3, digits=5, plotsize=default_plotsize,
                            legendpos=:topleft)
  binsC = bins
  binsT == nothing && (binsT = bins)
  histogram(yC[isfinite.(yC)], bins=binsC, label=L"y_C", alpha=alpha,
            color=:blue, linealpha=0, normalize=true)
  histogram!(yT[isfinite.(yT)], bins=binsT, label=L"y_T", alpha=alpha,
             color=:red, linealpha=0, normalize=true)
  p0C = round(mean(isinf.(yC)), digits=digits)
  p0T = round(mean(isinf.(yT)), digits=digits)
  title!("prop. -∞ in C: $(p0C) | prop. -∞ in T: $(p0T)", titlefont=font(10))
  plot!(size=plotsize, legend=:topleft)
  savefig(joinpath(imgdir, "data-hist.pdf"))
  closeall()
end


function postprocess(chain, laststate, summarystats, yC, yT, imgdir;
                     bw_postpred=0.2, density_legend_pos=:best,
                     ygrid=collect(range(-8, 8, length=1000)))
  # Print summary statistics.
  CDE.Model.printsummary(chain, summarystats)
  flush(stdout)

  # Print KS Statistic.
  ks_fit = rstats.ks_test(yC, yT)
  println("KS-test p-value: ", ks_fit["p.value"][1])
  flush(stdout)

  # Plot results.
  @time CDE.Model.plotpostsummary(chain, summarystats, yC, yT, imgdir,
                                  bw_postpred=bw_postpred, ygrid=ygrid,
                                  density_legend_pos=density_legend_pos)
  flush(stdout)
end

function postprocess(chain0, chain1, data, imgdir, awsbucket;
                     simdata=nothing, bw_postpred=0.20,
                     density_legend_pos=:best, binsC=nothing, binsT=nothing,
                     ygrid=collect(range(-8, 8, length=1000)))
  mkpath(imgdir)
  CDE.Util.redirect_stdout_to_file(joinpath(imgdir, "bf.txt")) do
    println("Start merging results...")

    # Compute Bayes factor in favor of M1.
    p = CDE.Model.group(:p, chain0)[1]
    @time lbf = CDE.Model.log_bayes_factor(data, VD(chain0[1]), VD(chain1[1]))
    pm1 = CDE.Model.posterior_prob1(lbf, logpriorodds=logit(p))
    println("Log Bayes Factor: $(lbf) | P(M1|y): $(pm1)")

    # Print KS Statistic.
    ks_fit = rstats.ks_test(data.yC, data.yT)
    println("KS-test p-value: ", ks_fit["p.value"][1])
    flush(stdout)

    # Plot marginal posterior density
    CDE.Model.plot_posterior_predictive(data.yC, data.yT, chain0, chain1, pm1,
                                        imgdir, bw_postpred=bw_postpred,
                                        density_legend_pos=density_legend_pos,
                                        simdata=simdata, lw=.5, ls=:solid, 
                                        binsC=binsC, binsT=binsT,
                                        xlims_=(-6, 6), digits=5, fontsize=7)

    CDE.Model.plot_gamma(data.yC, data.yT, chain0, chain1, pm1, imgdir)
  end

  # Send results
  if awsbucket != nothing
    CDE.Util.s3sync(from=imgdir, to=awsbucket, tags=`--exclude '*.nfs'`)
  end
end

function defaults(yC, yT, K; seed=nothing)
  seed == nothing || Random.seed!(seed)

  data = CDE.Model.Data(yC, yT)
  prior = CDE.Model.Prior(K, mu=CDE.Model.compute_prior_mu(data),
                          nu=LogNormal(1.6, 0.4), p=Beta(100, 100),
                          omega=InverseGamma(.1, .1), psi=Normal(-1, 10))
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
  simdata = config[:simdata]
  yC = simdata[:yC]
  yT = simdata[:yT]
  K = config[:K]
  nsamps = config[:nsamps]
  nburn = config[:nburn]
  thin = config[:thin]

  # Make image path id needed.
  imgdir = joinpath(resultsdir, "img")
  mkpath(imgdir)

  # Create initial state, data, prior, and tuners objects.
  state, data, prior, tuners = defaults(yC, yT, K, seed=0)
  state.p = config[:p]
  state.beta = config[:beta]
  println("Priors:")
  foreach(fn -> println(fn, " => ", getfield(prior, fn)),
          fieldnames(CDE.Model.Prior))

  # Plot data.
  plot_observed_hist(yC, yT, imgdir, bins=50, binsT=50*2,
                     alpha=0.3, digits=5, legendpos=:topleft)
  plot_true_data_density(simdata, imgdir, lw=1, legendpos=:topleft)
 
  # Run analysis.
  println("Run Chain ..."); flush(stdout)
  @time chain, laststate, summarystats = CDE.fit(
      state, data, prior, tuners, nsamps=[nsamps], fix=[:p, :beta],
      nburn=nburn, thin=thin, rep_aux=10)
  flush(stdout)

  # Save results
  BSON.bson("$(resultsdir)/results.bson",
            Dict(:chain=>chain, :laststate=>laststate, :data=>data,
                 :prior=>prior, :summarystats=>summarystats,
                 :simdata=>simdata))

  # Load data:
  out = BSON.load("$(resultsdir)/results.bson")

  # Post process
  postprocess(out[:chain], out[:laststate], out[:summarystats], out[:data].yC,
              out[:data].yT, imgdir; bw_postpred=0.3,
              ygrid=collect(range(-8, 8, length=1000)),
              density_legend_pos=:topleft)

  # Send results
  if awsbucket != nothing
    CDE.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
  end

  # Print done.
  println("Done!"); flush(stdout)
end
