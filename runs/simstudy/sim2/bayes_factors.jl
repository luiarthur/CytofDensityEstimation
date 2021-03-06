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
  snum = 4
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
K = 3
data = CDE.Model.Data(yC, yT)
prior_mu = let 
  yfinite = [data.yC_finite; data.yT_finite]
  Normal(mean(yfinite), std(yfinite))
end
prior = CDE.Model.Prior(K, mu=prior_mu, nu=LogNormal(1.6, 0.4), p=Beta(100, 100),
                        omega=InverseGamma(.1, .1), psi=Normal(-1, 1))
state = CDE.Model.State(data, prior)
tuners = CDE.Model.Tuners(K, 0.1)

println("Priors:")
for fn in fieldnames(CDE.Model.Prior)
  println(fn, " => ", getfield(prior, fn))
end

# Specify parameters to keep constant.
fix = Symbol[:beta, :p]

# Run chain.
state.p = 0.5

state.beta = false
@time chain0, laststate0, summarystats0 = CDE.Model.fit(
    state, data, prior, deepcopy(tuners), fix=fix,
    nsamps=[2000], nburn=2000, thin=1, rep_aux=10)

state.beta = true
@time chain1, laststate1, summarystats1 = CDE.Model.fit(
    state, data, prior, deepcopy(tuners), fix=fix,
    nsamps=[2000], nburn=2000, thin=1, rep_aux=10)

# Posterior Odds (in favor of treatment effect).
@time post_odds = CDE.Model.posterior_odds(data, chain0[1], chain1[1])
println("Posterior Odds: ", post_odds); flush(stdout)

imgdir0 = "$(resultsdir)/bf0/img"
mkpath(imgdir0)
@time CDE.Model.plotpostsummary(chain0, summarystats0, data.yC, data.yT, imgdir0)
flush(stdout)

imgdir1 = "$(resultsdir)/bf1/img"
mkpath(imgdir1)
@time CDE.Model.plotpostsummary(chain1, summarystats1, data.yC, data.yT, imgdir1)
flush(stdout)
