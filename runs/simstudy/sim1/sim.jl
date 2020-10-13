# TODO: DEBUG!
# - [ ] Fix some parameters, and see if I recover truth.
# - [ ] Plot posterior density.
# - [ ] Is it perhaps needed to sample the other parameters multiple times
#       after updating beta? Especially when the value of beta has changed?
# - [ ] Is it necessary to fit the model with fixed beta first?
# - [ ] Perhaps fit two models; one with beta=0, one with beta=1, 
#       then sample beta after somehow?
# - [ ] What happens if I don't marginalize over lambda when updating beta?

import Pkg; Pkg.activate("../../../")

using CytofDensityEstimation
using StatsPlots
using LaTeXStrings
using Distributions
using HypothesisTests
import Random

const CDE = CytofDensityEstimation

Random.seed!(1);
simdata = CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.2,
                                     etaC=[.5, .5, 0], etaT=[.5, .2, .3],
                                     loc=[-1, 1, 3.], scale=[.7, 1.3, 1],
                                     df=[15, 30, 10], skew=[-2, -5, 0.])

yC, yT = simdata[:yC], simdata[:yT]
legendfont=font(12)
density(yC[isfinite.(yC)],  lw=3, label=L"y_C", legendfont=legendfont,
        color=:blue)
density!(yT[isfinite.(yT)], lw=3, label=L"y_T", legendfont=legendfont,
         color=:red)

Random.seed!(0)
K = 5
data = CDE.Model.Data(yC, yT)
prior = CDE.Model.Prior(K)
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

chain, laststate, summarystats = CDE.Model.fit(state, data, prior, tuners,
                                               nsamps=[1000], nburn=2000)
gammaC = CDE.Model.group(:gammaC, chain)
gammaT = CDE.Model.group(:gammaT, chain)
mean(gammaC)
mean(gammaT)
mean(CDE.Model.group(:phi, chain))

etaC = CDE.Model.group(:etaC, chain)
etaT = CDE.Model.group(:etaT, chain)
mean(etaC)
mean(etaT)
mu = CDE.Model.group(:mu, chain)
mean(mu)

beta = CDE.Model.group(:beta, chain)
print("mean beta: ", mean(beta))

ll = [s[:loglike] for s in summarystats]
plot(ll)
plot(beta)

# ks_fit = ApproximateTwoSampleKSTest(yC, yT)

mean(gammaC)
mean(gammaT[beta .== 1])
