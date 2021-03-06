# NOTE: This simulation fits two models, and computes the posterior probability
# of β | y.

ENV["GKSwstype"] = "nul"  # For StatsPlots
println("pid: ", getpid())

# Read command line args.
if length(ARGS) > 1
  resultsdir = ARGS[1]
  awsbucket = ARGS[2]
  snum = parse(Int, ARGS[3])
  println("ARGS: ", ARGS)
else
  resultsdir = "results/test/"
  awsbucket = nothing
  snum = 3
end
flush(stdout)

# Create results/images dir if needed
imgdir = "$(resultsdir)/img"
mkpath(imgdir)

import Pkg; Pkg.activate(joinpath(@__DIR__, "../../../"))

using CytofDensityEstimation
const CDE = CytofDensityEstimation
using LaTeXStrings
using Distributions
using StatsPlots
using HypothesisTests
import Random
using RCall
using BSON
import CytofDensityEstimation.Model.default_ygrid
using StatsFuns

include("scenarios.jl")

plotsize = (400, 400)

# Simulate data.
Random.seed!(5);
simdata = scenarios(snum)
yC, yT = simdata[:yC], simdata[:yT]

# Plot true data density.
function plot_trude_data_density()
  plot(default_ygrid(), pdf.(simdata[:mmC], default_ygrid()), lw=3,
       label=L"y_C", color=:blue)
  plot!(default_ygrid(), pdf.(simdata[:mmT], default_ygrid()), lw=3,
        label=L"y_T", color=:red)
end
plot_trude_data_density()
plot!(size=plotsize)
savefig(joinpath(imgdir, "data-true-density.pdf"))
closeall()

# Plot observed data density.
function plot_observed_data_density()
  density(yC[isfinite.(yC)],  lw=3, label=L"y_C", color=:blue)
  density!(yT[isfinite.(yT)], lw=3, label=L"y_T", color=:red)
end
plot_observed_data_density()
plot!(size=plotsize)
savefig(joinpath(imgdir, "data-kde.pdf"))
closeall()

# Define data, prior, and initial state.
Random.seed!(7)
K = 5 # 3
data = CDE.Model.Data(yC, yT)
prior_mu = let 
  yfinite = [data.yC_finite; data.yT_finite]
  Normal(mean(yfinite), std(yfinite))
end
prior = CDE.Model.Prior(K, mu=prior_mu, nu=LogNormal(1.6, 0.4), 
                        p=Beta(1, 99), omega=InverseGamma(.1, .1),
                        eta=Dirichlet(K, 1 / K), psi=Normal(-1, 1))
state = CDE.Model.State(data, prior)
tuners = CDE.Model.Tuners(K, 0.1)

println("Priors:")
for fn in fieldnames(CDE.Model.Prior)
  println(fn, " => ", getfield(prior, fn))
end


function run(; p, beta)
  # Fix these parameters.
  state.p = p
  state.beta = beta

  # Run chain.
  @time chain, laststate, summarystats = CDE.fit(
      state, data, prior, deepcopy(tuners), nsamps=[3000], nburn=3000, thin=1,
      fix=[:p, :beta], rep_aux=10)

  # Make directoary for results if needed.
  resdir = "$(resultsdir)/m$(beta)"
  mkpath(resdir)

  # Save results
  BSON.bson("$(resdir)/results.bson",
            Dict(:chain=>deepcopy(chain), :laststate=>deepcopy(laststate),
                 :summarystats=>deepcopy(summarystats),
                 :simdata=>deepcopy(simdata)))

  return chain, laststate, summarystats
end

# Load rstats.
@rimport stats as rstats

function postprocess(beta)
  # Load results.
  out = BSON.load("$(resultsdir)/m$(beta)/results.bson")

  # Print summary statistics.
  CDE.Model.printsummary(out[:chain], out[:summarystats])
  flush(stdout)

  # Make imgdir if needed.
  _imgdir = joinpath(resultsdir, "m$(beta)/img")
  mkpath(_imgdir)

  # Plot results.
  @time CDE.Model.plotpostsummary(out[:chain], out[:summarystats], yC, yT,
                                  _imgdir, simdata=out[:simdata],
                                  bw_postpred=.15, xlims_=(-6, 6))
  flush(stdout)
end

# Run chains.
chain0, laststate0, summarystats0 = run(p=0.5, beta=0)
chain1, laststate1, summarystats1 = run(p=0.5, beta=1)

# Post process.
postprocess(0)
postprocess(1)

# Print KS statistic.
ks_fit = rstats.ks_test(yC, yT)
println("KS p-value: ", ks_fit["p.value"][1])
flush(stdout)

# Bayes Factor
@time lbf = CDE.Model.log_bayes_factor(data, chain0[1], chain1[1])
pm1 = CDE.Model.posterior_prob1(lbf, logpriorodds=logit(state.p))
println("Log Bayes Factor: $(lbf) | P(M1|y): $(pm1)")

# Plot marginal posterior density
CDE.Model.plot_posterior_predictive(yC, yT, chain0, chain1, pm1, imgdir,
                                    bw_postpred=0.15, simdata=simdata, lw=.5,
                                    ls=:solid, xlims_=(-6, 6), digits=5,
                                    fontsize=7)
CDE.Model.plot_gamma(yC, yT, chain0, chain1, pm1, imgdir)

# DIC
dic0, dic1 = CDE.dic(chain0[1], data), CDE.dic(chain1[1], data)
println("(DIC0, DIC1): ($(round(dic0, digits=3)), $(round(dic1, digits=3)))")
flush(stdout)

# Send results to S3.
if awsbucket != nothing
  CDE.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
end

println("Done!")
flush(stdout)
