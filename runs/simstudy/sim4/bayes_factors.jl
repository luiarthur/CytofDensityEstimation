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
imgdir0 = "$(resultsdir)/bf0/img"
imgdir1 = "$(resultsdir)/bf1/img"
mkpath(imgdir0)
mkpath(imgdir1)

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
savefig(joinpath(imgdir0, "data-true-density.pdf"))
savefig(joinpath(imgdir1, "data-true-density.pdf"))
closeall()

# Plot observed data density.
function plot_observed_data_density()
  density(yC[isfinite.(yC)],  lw=3, label=L"y_C", color=:blue)
  density!(yT[isfinite.(yT)], lw=3, label=L"y_T", color=:red)
end
plot_observed_data_density()
plot!(size=plotsize)
savefig(joinpath(imgdir0, "data-kde.pdf"))
savefig(joinpath(imgdir1, "data-kde.pdf"))
closeall()

# Define data, prior, and initial state.
Random.seed!(7)
K = 3
data = CDE.Model.Data(yC, yT)
prior_mu = let 
  yfinite = [data.yC_finite; data.yT_finite]
  Normal(mean(yfinite), std(yfinite))
end
prior = CDE.Model.Prior(K, mu=prior_mu, nu=LogNormal(1.6, 0.4),
                        p=Beta(100, 100), omega=InverseGamma(.1, .1),
                        psi=Normal(-1, 1))
state = CDE.Model.State(data, prior)
tuners = CDE.Model.Tuners(K, 0.1)

println("Priors:")
for fn in fieldnames(CDE.Model.Prior)
  println(fn, " => ", getfield(prior, fn))
end

# Specify parameters to keep constant.
fix = Symbol[:beta, :p]

function fit_model(p, beta)
  statej = deepcopy(state)
  tunersj = deepcopy(tuners)
  statej.p = p
  statej.beta = beta
  return CDE.Model.fit(statej, data, prior, tunersj, fix=fix, nsamps=[2000],
                       nburn=4000, thin=1, rep_aux=10)
end

# Run chain0.
chain0, laststate0, summarystats0 = fit_model(0.5, false)
BSON.bson("$(resultsdir)/bf0/results.bson",
          Dict(:chain=>chain0, :laststate=>laststate0,
               :summarystats=>summarystats0, :simdata=>simdata))

# Run chain1.
chain1, laststate1, summarystats1 = fit_model(0.5, true)
BSON.bson("$(resultsdir)/bf1/results.bson",
          Dict(:chain=>chain1, :laststate=>laststate1,
               :summarystats=>summarystats1, :simdata=>simdata))

# Read results.
out0 = BSON.load("$(resultsdir)/bf0/results.bson")
out1 = BSON.load("$(resultsdir)/bf1/results.bson")

# Bayes Factor.
VD = Vector{Dict{Symbol, Any}}
@time lbf = CDE.Model.log_bayes_factor(data,
                                       VD(out0[:chain][1]), VD(out1[:chain][1]))
pp1 = CDE.Model.posterior_prob1(lbf)
println("Log Bayes Factor: $(lbf) | Post. prob. for model 1: $(pp1)")

# DIC
dic0 = CDE.dic(VD(out0[:chain][1]), data)
dic1 = CDE.dic(VD(out1[:chain][1]), data)
println("(DIC0, DIC1): ($(round(dic0, digits=3)), $(round(dic1, digits=3)))")
println("Model ", Int(dic0 > dic1), " is preferred.")

# Save figures 0
@time CDE.Model.plotpostsummary(out0[:chain], out0[:summarystats],
                                data.yC, data.yT, imgdir0,
                                simdata=out0[:simdata], bw_postpred=.2,
                                xlims_=(-6, 6))
flush(stdout)

# Save figures 1
@time CDE.Model.plotpostsummary(out1[:chain], out1[:summarystats],
                                data.yC, data.yT, imgdir1,
                                simdata=out1[:simdata], bw_postpred=.2,
                                xlims_=(-6, 6))
flush(stdout)

if awsbucket != nothing
  CDE.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
end
println("Done!")
