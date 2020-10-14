# TODO: DEBUG!
# - [ ] Fix some parameters, and see if I recover truth.
# - [ ] Plot posterior density.
# - [ ] Is it perhaps needed to sample the other parameters multiple times
#       after updating beta? Especially when the value of beta has changed?
# - [ ] Is it necessary to fit the model with fixed beta first?
# - [ ] Perhaps fit two models; one with beta=0, one with beta=1, 
#       then sample beta after somehow?
# - [ ] What happens if I don't marginalize over lambda when updating beta?
# - [ ] If beta goes from 0 -> 1, update all other parameters 50-100 times.
# - [ ] Or if beta goes from 1 -> 0, stop updating gammaT and etaT.
# - Issues: cannot recover mu. Seems to be related to (v, zeta).
# - NOTE: nu is a trouble-maker. Fixing it solves most of the problems!!! 

import Pkg; Pkg.activate("../../../")

using CytofDensityEstimation
using StatsPlots
using LaTeXStrings
using Distributions
using HypothesisTests
import Random
import StatsPlots.KernelDensity.kde

const CDE = CytofDensityEstimation

Random.seed!(2);
simdata = CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.2,
                                     etaC=[.5, .5, 0], etaT=[.5, .2, .3],
                                     # etaC=[.5, .5, 0], etaT=[.5, .5, .0],
                                     loc=[-1, 1, 3.], scale=[1, 1, 1],
                                     df=[15, 30, 10], skew=[-20, -5, 0.])

yC, yT = simdata[:yC], simdata[:yT]
legendfont=font(12)
density(yC[isfinite.(yC)],  lw=3, label=L"y_C", legendfont=legendfont,
        color=:blue)
density!(yT[isfinite.(yT)], lw=3, label=L"y_T", legendfont=legendfont,
         color=:red)

Random.seed!(1)
K = 7
data = CDE.Model.Data(yC, yT)
prior_mu = let 
  yfinite = [data.yC_finite; data.yT_finite]
  Normal(mean(yfinite), std(yfinite)*3)
end
prior = CDE.Model.Prior(K, mu=prior_mu, nu=LogNormal(3, .01), p=Beta(.1, .9))
state = CDE.Model.State(data, prior)
tuners = CDE.Model.Tuners(K)

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
fix=Symbol[]
# fix=Symbol[:nu]

flags=Symbol[:update_beta_with_skewt, :update_lambda_with_skewt]
# flags=Symbol[]  # TODO: Test this case.

init = deepcopy(state)
chain, laststate, summarystats = CDE.Model.fit(init, data, prior, tuners,
                                               nsamps=[1000], nburn=1000,
                                               fix=fix, flags=flags)
# v, zeta, lambda

# ks_fit = ApproximateTwoSampleKSTest(yC, yT)

ll = [s[:loglike] for s in summarystats]
gammaC = CDE.Model.group(:gammaC, chain)
gammaT = CDE.Model.group(:gammaT, chain)
phi = CDE.Model.group(:phi, chain)
etaC = CDE.Model.group(:etaC, chain)
etaT = CDE.Model.group(:etaT, chain)
mu = CDE.Model.group(:mu, chain)
beta = CDE.Model.group(:beta, chain)
sigma = CDE.Model.group(:sigma, chain)
phi = CDE.Model.group(:phi, chain)
nu = CDE.Model.group(:nu, chain)
psi = CDE.Model.group(:psi, chain)
omega = CDE.Model.group(:omega, chain)

CDE.Model.printsummary(chain, summarystats)

ygrid = range(-6, 6, length=1000)
post_dens = CDE.Model.posterior_density(chain, ygrid)

pdfC_lower = CDE.Model.MCMC.quantiles(hcat(post_dens[1]...), .025, dims=2)
pdfC_upper = CDE.Model.MCMC.quantiles(hcat(post_dens[1]...), .975, dims=2)
pdfT_lower = CDE.Model.MCMC.quantiles(hcat(post_dens[2]...), .025, dims=2)
pdfT_upper = CDE.Model.MCMC.quantiles(hcat(post_dens[2]...), .975, dims=2)
pdfC_mean = mean(post_dens[1])
pdfT_mean = mean(post_dens[2])

# Plot density
plot(kde(yC[isfinite.(yC)], bandwidth=.3), lw=3, label=L"y_C",
     legendfont=legendfont, color=:blue, ls=:dot)
plot!(kde(yT[isfinite.(yT)], bandwidth=.3), lw=3, label=L"y_T",
     legendfont=legendfont, color=:red, ls=:dot)
plot!(ygrid, pdfC_mean, ribbon=(pdfC_mean-pdfC_lower, pdfC_upper-pdfC_mean),
     color=:blue)
plot!(ygrid, pdfT_mean, ribbon=(pdfT_mean-pdfT_lower, pdfT_upper-pdfT_mean),
      color=:red)
