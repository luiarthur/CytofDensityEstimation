function loglikeam(data::Data, s::Dict{Symbol, Any}, p::Float64)
  state = StateAM(s[:gammaC], s[:gammaT],
                  s[:etaC], s[:etaT], s[:mu], s[:sigma],
                  s[:nu], s[:phi])
  return loglike(state, data, p)
end

function loglike(s::StateAM, data::Data, p::Float64)
  return loglike_C(s, data) + loglike_T(s, data, p)
end

function loglike_C(s::StateAM, data::Data)
  f = Util.mixskewtlogpdf(s.mu', s.sigma', s.nu', s.phi', s.etaC',
                          data.yC_finite, dims=2)
  return data.ZC * log(s.gammaC) + data.NC_finite * log1p(-s.gammaC) + sum(f)
end

function loglike_T(s::StateAM, data::Data, p)
  f = Util.skewtlogpdf.(s.mu', s.sigma', s.nu', s.phi', data.yT_finite)
  fC = sum(logsumexp(f .+ log.(s.etaC'), dims=2))
  fT = sum(logsumexp(f .+ log.(s.etaT'), dims=2))
  gT = data.ZT * log(s.gammaT) + data.NT_finite * log1p(-s.gammaT)
  gC = data.ZT * log(s.gammaC) + data.NT_finite * log1p(-s.gammaC)
  return logsumexp([fT + gT + log(p), fC + gC + log1p(-p)])
end
