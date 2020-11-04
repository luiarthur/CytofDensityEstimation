module CytofDensityEstimation

#= NOTE:
Julia uses Gamma(shape, scale) and InverseGamma(shape, scale).
Note that InverseGamma(shape, scale) IS my preferred parameterization.
It has a mean of scale / (shape - 1) for shape > 1.
=#

include("Util/Util.jl")
include("MCMC/MCMC.jl")
include("Model/Model.jl")

using .Model

end # module
