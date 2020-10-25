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

# Run chain.
state.beta = 0
@time chain, laststate, summarystats = CDE.ppfit(state, data, prior, tuners,
                                                 p=0.5, nsamps=[5000],
                                                 nburn=1000, thin=1,
                                                 rep_aux=10, warmup=1000)
    

# Save results
BSON.bson("$(resultsdir)/results.bson",
          Dict(:chain=>chain, :laststate=>laststate,
               :summarystats=>summarystats, :simdata=>simdata))

# Load via:
out = BSON.load("$(resultsdir)/results.bson")

# Print KS statistic.
@rimport stats as rstats
ks_fit = rstats.ks_test(out[:simdata][:yC], out[:simdata][:yT])
println(ks_fit["p.value"][1])
flush(stdout)

# Print summary statistics.
CDE.Model.printsummary(out[:chain], out[:summarystats])
flush(stdout)

# Plot results.
@time CDE.Model.plotpostsummary(out[:chain], out[:summarystats], yC, yT,
                                imgdir, simdata=out[:simdata], bw_postpred=.2,
                                xlims_=(-6, 6))
flush(stdout)

if awsbucket != nothing
  CDE.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
end

println("Done!")
flush(stdout)

# Bayes Factor
VD = Vector{Dict{Symbol, Any}}
chain1 = filter(s -> s[:beta] == 1, out[:chain][1])
chain0 = filter(s -> s[:beta] == 0, out[:chain][1])
@time if length(chain1) > 0 && length(chain0) > 0
  @time bf = CDE.Model.bayes_factor(data, VD(chain0), VD(chain1))
  pp1 = CDE.Model.posterior_prob1(bf)
  println("Bayes Factor: $(bf) | Post. prob. for model 1: $(pp1)")
else
  if length(chain1) > 0
    println("M1 is preferred.")
  else
    println("M0 is preferred.")
  end
end
flush(stdout)

# DIC
@time dic0, dic1 = CDE.dic(VD(chain0), data), CDE.dic(VD(chain1), data)
println("(DIC0, DIC1): ($(round(dic0, digits=3)), $(round(dic1, digits=3)))")
