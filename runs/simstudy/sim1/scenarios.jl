#=
import Pkg; Pkg.activate(joinpath(@__DIR__, "../../../"))
using CytofDensityEstimation
const CDE = CytofDensityEstimation
=#


function scenarios(n)
  if n == 1
    return CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.2,
                                      etaC=[.5, .5, 0], etaT=[.5, .2, .3],
                                      loc=[-1, 1, 3.], scale=[.7, .7, .7],
                                      df=[15, 30, 10], skew=[-20, -5, 0.])
  elseif n == 2
    return CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.3,
                                      etaC=[.5, .5, 0], etaT=[.5, .2, .3],
                                      loc=[-1, 1, 3.], scale=[.7, .7, .7],
                                      df=[15, 30, 10], skew=[-20, -5, 0.])
  elseif n == 3
    return CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.3,
                                      etaC=[.5, .5, 0], etaT=[.5, .5, .0],
                                      loc=[-1, 1, 3.], scale=[.7, .7, .7],
                                      df=[15, 30, 10], skew=[-20, -5, 0.])
  end
end
