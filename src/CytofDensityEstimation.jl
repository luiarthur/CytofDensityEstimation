module CytofRepFAM

#= NOTE:
Julia uses Gamma(shape, scale) and InverseGamma(shape, scale).
Note that InverseGamma(shape, scale) IS my preferred parameterization.
It has a mean of scale / (shape - 1) for shape > 1.
=#

include("Util/Util.jl")
include("Model/Model.jl")

end # module
