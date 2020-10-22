# TODO: DEBUG!
# - [ ] Is it perhaps needed to sample the other parameters multiple times
#       after updating beta? Especially when the value of beta has changed?
# - [ ] Is it necessary to fit the model with fixed beta first?
# - [ ] Perhaps fit two models; one with beta=0, one with beta=1, 
#       then sample beta after somehow?
# - [ ] What happens if I don't marginalize over lambda when updating beta?
# - NOTE: nu is a trouble-maker. Fixing it solves most of the problems!!! 

ENV["GKSwstype"] = "nul"  # For StatsPlots

import Pkg; Pkg.activate("../../../")

using CytofDensityEstimation
using LaTeXStrings
using Distributions
using StatsPlots
using HypothesisTests
import Random
using BSON

const CDE = CytofDensityEstimation

resultsdir = "results"
imgdir = "$(resultsdir)/img"
mkpath(imgdir)

Random.seed!(2);
simdata = CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.2,
                                     etaC=[.5, .5, 0], etaT=[.5, .2, .3],
                                     # etaC=[.5, .5, 0], etaT=[.5, .5, .0],
                                     loc=[-1, 1, 3.], scale=[1, 1, 1]/10,
                                     df=[15, 30, 10], skew=[-20, -5, 0.])
yC, yT = simdata[:yC], simdata[:yT]

# Plot data.
density(yC[isfinite.(yC)],  lw=3, label=L"y_C", color=:blue)
density!(yT[isfinite.(yT)], lw=3, label=L"y_T", color=:red)
savefig(joinpath(imgdir, "data.pdf"))
closeall()

Random.seed!(1)
K = 6
data = CDE.Model.Data(yC, yT)
prior_mu = let 
  yfinite = [data.yC_finite; data.yT_finite]
  Normal(mean(yfinite), std(yfinite)*3)
end
prior = CDE.Model.Prior(K, mu=prior_mu, nu=LogNormal(3, .01), p=Beta(.1, .9))
state = CDE.Model.State(data, prior)
tuners = CDE.Model.Tuners(K)

println("Priors:\n", prior)

# state.beta=0
# chain0, laststate, summarystats = CDE.Model.fit(state, data, prior, tuners,
#                                                 nsamps=[1000], nburn=2000,
#                                                 fix=[:beta])
# 
# state.beta=1
# chain1, laststate, summarystats = CDE.Model.fit(state, data, prior, tuners,
#                                                 nsamps=[1000], nburn=2000,
#                                                 fix=[:beta])

# state.beta = true
# state.gammaC = simdata[:gammaC]
# state.gammaT = simdata[:gammaT]
# state.etaC .= [simdata[:etaC]; zeros(2)]
# state.etaT .= [simdata[:etaT]; zeros(2)]
# state.mu .= [simdata[:loc]; rand(prior.mu, 2)]
# state.nu .= [simdata[:df]; rand(prior.nu, 2)]
# state.vC .= 1
# state.vT .= 1
# state.psi .= let
#   _psi = CDE.Util.toaltskew.(simdata[:scale], simdata[:skew])
#   [_psi; rand(prior.psi, 2)]
# end
# state.omega .= let
#   _omega = CDE.Util.toaltscale.(simdata[:scale], simdata[:skew]) .^ 2
#   [_omega; rand(prior.omega, 2)]
# end

# fix=[:gammaC, :gammaT, :mu, :nu, :omega, :psi, :etaC, :etaT]
# fix=[:gammaC, :gammaT, :mu, :nu, :omega, :psi, :v]
# fix=Symbol[]  # TODO: Test this case.
# fix=[:gammaC, :gammaT, :nu, :omega, :psi, :etaC, :etaT]
# fix=[:gammaC, :gammaT, :etaC, :etaT, :nu]
# fix=[:gammaC, :gammaT, :etaC, :etaT, :v]
# fix=[:v]
# fix=Symbol[]
fix=Symbol[:nu]

init = deepcopy(state)
chain, laststate, summarystats = CDE.Model.fit(init, data, prior, tuners,
                                               nsamps=[1000], nburn=1000,
                                               fix=fix)

# Save results
BSON.bson("$(resultsdir)/results.bson",
          Dict(:chain=>chain, :laststate=>laststate, :summarystats=>summarystats))

# Load via:
# using BSON, CytofDensityEstimation
out = BSON.load("$(resultsdir)/results.bson")
chain = out[:chain]
laststate = out[:laststate]
summarystats = out[:summarystats]

ks_fit = ApproximateTwoSampleKSTest(yC, yT)
println(ks_fit)

CDE.Model.printsummary(chain, summarystats)

@time CDE.Model.plotpostsummary(chain, summarystats, yC, yT, imgdir,
                                simdata=simdata, bw_postpred=.1)

# Hard to recover (for K≥3):
# simdata = CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.2,
#                                      etaC=[.5, .5, 0], etaT=[.5, .2, .3],
#                                      loc=[-1, 1, 3.], scale=[1, 1, 1]/10,
#                                      df=[15, 30, 10], skew=[-20, -5, 0.])
# - nu, phi (psi). All other parameters are easily recovered.
