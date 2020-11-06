module Model

using Distributions
using StatsFuns
import Printf

# For plotting.
using StatsPlots, LaTeXStrings
import StatsPlots.KernelDensity.kde
import LinearAlgebra

include("../Util/Util.jl")
using .Util: SkewT

include("../MCMC/MCMC.jl")
include("Simulate.jl")
include("Data.jl")
include("Prior.jl")
include("State.jl")
include("Tuners.jl")
include("Flag.jl")
include("loglike.jl")
include("updates/update.jl")
include("fit.jl")

include("PriorAM.jl")
include("StateAM.jl")
include("logprioram.jl")
include("loglikeam.jl")
include("amfit.jl")

include("logprior.jl")
include("pseudo_prior_updates/update.jl")
include("fit_via_pseudo_prior.jl")
include("postprocess.jl")

export UpdateBetaWithSkewT, UpdateLambdaWithSkewT, fit, amfit
export fit_via_pseudo_prior, ppfit, posterior_prob1, bayes_factor, dic

end # module
