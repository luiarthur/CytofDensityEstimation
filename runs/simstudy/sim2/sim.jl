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

# Parameters to fix
# - psi is near truth when mu, omega, nu, and eta are fixed, and n → ∞.
# - nu can be recovered when mu, omega, psi, and eta are fixed, and n → ∞.
# - mu is easy to recover when nu, omega, psi, and eta are fixed.
# - omega is near truth when mu, nu, psi, and eta are fixed; it requires a
#   large amount of data to estimate.
# - eta can be easily recovered when n → ∞.
fix = Symbol[]
# fix = Symbol[:omega]
# fix = Symbol[:psi]
# fix = Symbol[:nu, :psi]
# fix = Symbol[:mu, :omega, :nu, :eta]
# state.mu[1:3] .= simdata[:loc]
# state.nu[1:3] .= simdata[:df]
# state.psi[1:3] .= CDE.Util.toaltskew.(simdata[:scale], simdata[:skew])
# state.omega[1:3] .= CDE.Util.toaltscale.(simdata[:scale], simdata[:skew]) .^ 2
# state.etaC .= 0; state.etaC[:1:3] .= simdata[:etaC]
# state.etaT .= 0; state.etaT[:1:3] .= simdata[:etaT]
monitors = CDE.Model.default_monitors()

# Warmup.
# _tuners = deepcopy(tuners)
# state.beta = 1
# @time _, state, _ = CDE.Model.fit(
#     state, data, prior, _tuners, nsamps=[1], nburn=200, thin=1,
#     fix=[fix; [:beta]], monitors=monitors,
#     rep_beta_flipped=50)

# Run chain.
@time chain, laststate, summarystats = CDE.Model.fit(
    state, data, prior, tuners, nsamps=[10000], nburn=2000, thin=1,
    rep_aux=10,
    fix=fix, monitors=monitors, rep_beta_flipped=50)

# Save results
BSON.bson("$(resultsdir)/results.bson",
          Dict(:chain=>chain, :laststate=>laststate,
               :summarystats=>summarystats, :simdata=>simdata))

# Load via:
# using BSON, CytofDensityEstimation
out = BSON.load("$(resultsdir)/results.bson")

# Print KS statistic.
# ks_fit = ApproximateTwoSampleKSTest(yC, yT)
# println(ks_fit)
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
