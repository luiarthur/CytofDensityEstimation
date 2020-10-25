#=
import Pkg; Pkg.activate(joinpath(@__DIR__, "../../../"))
using CytofDensityEstimation
const CDE = CytofDensityEstimation
=#


function scenarios(n::Int)
  # NOTE: Later.
  # loc = [-1, 0., 1]
  # scale = [1, 1, 1]

  loc = [-1, 1, 3.]
  scale = [.7, .7, .7]
  etaC = [.5, .5, 0]
  df = [7, 5, 10.]
  skew = [-5, -3, 0.]
  Ni = 1000

  if n == 0  # TEST CASE!
    return CDE.Model.generate_samples(NC=100, NT=100, gammaC=0.3, gammaT=0.3,
                                      etaC=[1., 0, 0], etaT=[1., 0, 0],
                                      loc=loc, scale=scale, df=df, skew=skew)
  elseif n == 1
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.3, gammaT=0.2,
                                      etaC=etaC, etaT=[.5, .4, .1],
                                      loc=loc, scale=scale, df=df, skew=skew)
  elseif n == 2
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.3, gammaT=0.3,
                                      etaC=etaC, etaT=[.5, .4, .1],
                                      loc=loc, scale=scale, df=df, skew=skew)
  elseif n == 3
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.3, gammaT=0.3,
                                      etaC=etaC, etaT=[.5, .45, .05],
                                      loc=loc, scale=scale, df=df, skew=skew)
  elseif n == 4
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.3, gammaT=0.3,
                                      etaC=etaC, etaT=[.5, .5, .0],
                                      loc=loc, scale=scale, df=df, skew=skew)
  else
    throw(ArgumentError("n=$(n) was provided. But n must be an integer between 0 and 5."))
  end
end
