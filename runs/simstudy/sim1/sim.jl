import Pkg; Pkg.activate("../../../")

using CytofDensityEstimation
using StatsPlots
using LaTeXStrings
using Distributions
import Random

const CDE = CytofDensityEstimation

Random.seed!(1);
simdata = CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.3,
                                     etaC=[.5, .5, 0], etaT=[.5, .3, .2],
                                     loc=[-1, 1, 2.], scale=[.7, 1.3, 1],
                                     df=[15, 30, 10], skew=[-2, -5, 0.])

yC, yT = simdata[:yC], simdata[:yT]
legendfont=font(12)
density(yC[isfinite.(yC)],  lw=3, label=L"y_C", legendfont=legendfont, color=:blue)
density!(yT[isfinite.(yT)], lw=3, label=L"y_T", legendfont=legendfont, color=:red)


Random.seed!(2)
K = 5
data = CDE.Model.Data(yC, yT)
prior = CDE.Model.Prior(K, p=Beta(.5, .5))
state = CDE.Model.State(data, prior)
tuners = CDE.Model.Tuners(K)
chain, laststate, summarystats = CDE.Model.fit(state, data, prior, tuners,
                                               nsamps=[1000], nburn=1000);

mean(CDE.Model.group(:beta, chain))
mean(CDE.Model.group(:gammaC, chain))
mean(CDE.Model.group(:gammaT, chain))
mean(CDE.Model.group(:sigma, chain))
mean(CDE.Model.group(:phi, chain))

mean(CDE.Model.group(:etaC, chain))
mean(CDE.Model.group(:etaT, chain))


