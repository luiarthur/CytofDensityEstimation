function scenarios(n::Int; seed=nothing, Ni=1000)
  seed == nothing || Random.seed!(seed)

  if n == 0  # TEST CASE!
    return scenarios(1, seed=seed, Ni=100)
  elseif n == 1
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.1, gammaT=0.1,
                                      etaC=[.25, .75, 0, 0],
                                      etaT=[.05, .05, .9, 0], 
                                      loc = [-1.5, 3.5, 5.1, 5],
                                      scale = [1.6, 1.76, 1.76, 1.6],
                                      df = [12, 10, 10, 15.],
                                      skew = [0, -10, -10, 0.])
  elseif n == 2
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.1, gammaT=0.1,
                                      etaC=[.05, .05, .5, .4], 
                                      etaT=[.05, .05, .9, 0], 
                                      loc = [-1.5, 3.5, 5.1, 4.3],
                                      scale = [1.6, 1.76, 1.76, 1.6],
                                      df = [12, 10, 10, 15.],
                                      skew = [0, -10, -10, -11.])
  elseif n == 3
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.1, gammaT=0.05,
                                      etaC=[.05, .05, .5, .4], 
                                      etaT=[.05, .05, .5, .4], 
                                      loc = [-1.5, 3.5, 5.1, 4.3],
                                      scale = [1.6, 1.76, 1.76, 1.6],
                                      df = [12, 10, 10, 15.],
                                      skew = [0, -10, -10, -11.])
  elseif n == 4
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.1, gammaT=0.1,
                                      etaC=[.05, .05, .5, .4], 
                                      etaT=[.05, .05, .5, .4], 
                                      loc = [-1.5, 3.5, 5.1, 4.3],
                                      scale = [1.6, 1.76, 1.76, 1.6],
                                      df = [12, 10, 10, 15.],
                                      skew = [0, -10, -10, -11.])
  elseif n == 5  # a test case
    return CDE.Model.generate_samples(NC=Ni, NT=Ni, gammaC=0.1, gammaT=0.1,
                                      etaC=[0, 1., 0, 0],
                                      etaT=[0, 0, 1., 0],
                                      loc = [-1.5, 3.5, 5.1, 4.3],
                                      scale = [1.6, 1.76, 1.76, 1.6],
                                      df = [12, 10, 10, 15.],
                                      skew = [0, -10, -10, -11.])
  else
    throw(ArgumentError("n=$(n) was provided. But n must be an integer between 0 and 5."))
  end
end
