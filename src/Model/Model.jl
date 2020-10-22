module Model

using Distributions
using StatsFuns
import Printf

# For plotting.
using StatsPlots, LaTeXStrings
import StatsPlots.KernelDensity.kde

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

include("logprior.jl")
include("pseudo_prior_updates/update.jl")
include("fit_via_pseudo_prior.jl")
include("postprocess.jl")

export UpdateBetaWithSkewT, UpdateLambdaWithSkewT, fit
export fit_via_pseudo_prior, ppfit

end # module
