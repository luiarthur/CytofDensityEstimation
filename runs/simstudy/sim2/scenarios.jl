#=
import Pkg; Pkg.activate(joinpath(@__DIR__, "../../../"))
using CytofDensityEstimation
const CDE = CytofDensityEstimation
=#


function scenarios(n)
  # NOTE: Later.
  # loc = [-1, 0., 1]
  # scale = [1, 1, 1]

  loc = [-1, 1, 3.]
  scale = [.7, .7, .7]
  etaC = [.5, .5, 0]
  df = [3, 5, 3.]
  skew = [-10, -5, 0.]
  if n == 1
    return CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.2,
                                      etaC=etaC, etaT=[.5, .4, .1],
                                      loc=loc, scale=scale,
                                      df=df, skew=skew)
  elseif n == 2
    return CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.3,
                                      etaC=etaC, etaT=[.5, .4, .1],
                                      loc=loc, scale=scale,
                                      df=df, skew=skew)
  elseif n == 3
    return CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.3,
                                      etaC=etaC, etaT=[.5, .45, .05],
                                      loc=loc, scale=scale,
                                      df=df, skew=skew)
  elseif n == 4
    return CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.3,
                                      etaC=etaC, etaT=[.5, .5, .0],
                                      loc=loc, scale=scale,
                                      df=df, skew=skew)
  elseif n == 5
    return CDE.Model.generate_samples(NC=1000, NT=1000, gammaC=0.3, gammaT=0.3,
                                      etaC=etaC, etaT=[1., 0, 0],
                                      loc=loc, scale=scale,
                                      df=df, skew=skew)
  end
end
