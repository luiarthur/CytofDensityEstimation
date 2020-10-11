module MCMC

using Distributions
using StatsFuns

include("TuningParam.jl")
include("metropolis.jl")
include("misc.jl")
include("gibbs.jl")

end # module
