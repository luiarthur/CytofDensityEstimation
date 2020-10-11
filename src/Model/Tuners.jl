struct Tuners{F <: AbstractFloat}
  nu::Vector{MCMC.TuningParam{F}}
end


function Tuners(K, value=1.0)
  nu_tuner = [MCMC.TuningParam(value) for _ in 1:K]
  return Tuners(nu_tuner)
end
