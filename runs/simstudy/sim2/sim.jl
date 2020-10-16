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
  snum = 2
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
import CytofDensityEstimation.Model.default_ygrid

include("scenarios.jl")

# Simulate data.
Random.seed!(2);
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
savefig(joinpath(imgdir, "data-density.pdf"))
closeall()

# Plot observed data density.
function plot_observed_data_density()
  density(yC[isfinite.(yC)],  lw=3, label=L"y_C", color=:blue)
  density!(yT[isfinite.(yT)], lw=3, label=L"y_T", color=:red)
end
plot_observed_data_density()
savefig(joinpath(imgdir, "data.pdf"))
closeall()

# Define data, prior, and initial state.
Random.seed!(5)
K = 3
data = CDE.Model.Data(yC, yT)
prior_mu = let 
  yfinite = [data.yC_finite; data.yT_finite]
  Normal(mean(yfinite), std(yfinite))
end
prior = CDE.Model.Prior(K, mu=prior_mu, nu=LogNormal(2, 1), p=Beta(.1, .9),
                        omega=InverseGamma(3, 2), psi=Normal(-2, 3))
state = CDE.Model.State(data, prior)
tuners = CDE.Model.Tuners(K, 0.1)

println("Priors:")
for fn in fieldnames(CDE.Model.Prior)
  println(fn, " => ", getfield(prior, fn))
end

# Specify flags for modeling.
flags = Symbol[:update_beta_with_skewt, :update_lambda_with_skewt]
# flags = Symbol[:update_lambda_with_skewt]
# flags = Symbol[]
# flags = Symbol[]

# Parameters to fix
fix = Symbol[]
# fix = Symbol[:mu, :nu, :omega]
# state.mu[1:3] .= simdata[:loc]
# state.nu[1:3] .= simdata[:df]
# state.psi[1:3] .= CDE.Util.toaltskew.(simdata[:scale], simdata[:skew])
# state.omega[1:3] .= CDE.Util.toaltscale.(simdata[:scale], simdata[:skew]) .^ 2

# fix = Symbol[:psi]
# fix = Symbol[:mu, :psi, :omega]
# state.mu .= simdata[:loc]
# state.psi .= CDE.Util.toaltskew.(simdata[:scale], simdata[:skew])
# state.omega .= CDE.Util.toaltscale.(simdata[:scale], simdata[:skew]).^2
# fix = Symbol[:nu, :vC, :vT, :eta, :mu]
# fix = Symbol[:mu, :nu, :vC, :vT, :omega]
# state.vC .= 1
# state.vT .= 1
# state.zetaC .= 1
# state.zetaT .= 1
# state.etaC .= simdata[:etaC]
# state.etaT .= simdata[:etaT]
# state.mu .= simdata[:loc]
# state.omega  .= 0.7^2

# monitors=[[CDE.Model.default_monitors()[1];
#            [:vC, :vT, :zetaC, :zetaT, :lambdaC, :lambdaT]]]
monitors = CDE.Model.default_monitors()

# Run chain
@time chain, laststate, summarystats = CDE.Model.fit(
    state, data, prior, tuners, nsamps=[2000], nburn=1000, thin=1,
    fix=fix, flags=flags, monitors=monitors)

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
@time CDE.Model.plotpostsummary(out[:chain], out[:summarystats], yC, yT,
                                imgdir, simdata=out[:simdata], bw_postpred=.2,
                                xlims_=(-6, 6))
flush(stdout)

if awsbucket != nothing
  CDE.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
end

println("Done!")
flush(stdout)
