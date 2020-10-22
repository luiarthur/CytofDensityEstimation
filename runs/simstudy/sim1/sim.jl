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
  snum = 1
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
using BSON

include("scenarios.jl")

# Simulate data.
Random.seed!(2);
simdata = scenarios(snum)
yC, yT = simdata[:yC], simdata[:yT]

# Plot data.
density(yC[isfinite.(yC)],  lw=3, label=L"y_C", color=:blue)
density!(yT[isfinite.(yT)], lw=3, label=L"y_T", color=:red)
savefig(joinpath(imgdir, "data.pdf"))
closeall()

# Define data, prior, and initial state.
Random.seed!(1)
K = 6
data = CDE.Model.Data(yC, yT)
prior_mu = let 
  yfinite = [data.yC_finite; data.yT_finite]
  Normal(mean(yfinite), std(yfinite)*2)
end
prior = CDE.Model.Prior(K, mu=prior_mu, nu=LogNormal(4, .01), p=Beta(.1, .9))
state = CDE.Model.State(data, prior)
tuners = CDE.Model.Tuners(K)

println("Priors:")
for fn in fieldnames(CDE.Model.Prior)
  println(fn, " => ", getfield(prior, fn))
end

# Parameters to fix
fix = Symbol[]

# Run chain
init = deepcopy(state)
@time chain, laststate, summarystats = CDE.Model.fit(init, data, prior, tuners,
                                                     nsamps=[1000], nburn=1000,
                                                     fix=fix)

# Save results
BSON.bson("$(resultsdir)/results.bson",
          Dict(:chain=>chain, :laststate=>laststate,
               :summarystats=>summarystats, :simdata=>simdata))

# Load via:
# using BSON, CytofDensityEstimation
out = BSON.load("$(resultsdir)/results.bson")

# Print KS statistic.
ks_fit = ApproximateTwoSampleKSTest(yC, yT)
println(ks_fit)
flush(stdout)

# Print summary statistics.
CDE.Model.printsummary(out[:chain], out[:summarystats])
flush(stdout)

# Plot results.
@time CDE.Model.plotpostsummary(out[:chain], out[:summarystats], yC, yT, imgdir,
                                simdata=out[:simdata], bw_postpred=.1)
flush(stdout)

if awsbucket != nothing
  CDE.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
end

println("Done!")
flush(stdout)
