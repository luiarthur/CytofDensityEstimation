function loglikeam(data::Data, s::Dict{Symbol, Any}, beta::Bool)
  state = StateAM(s[:gammaC], s[:gammaT],
                  s[:etaC], s[:etaT], s[:mu], s[:sigma],
                  s[:nu], s[:phi])
  return loglike(state, data, beta)
end

function loglike(s::StateAM, data::Data, beta::Bool)
  return loglike_C(s, data) + loglike_T(s, data, beta)
end

function loglike_C(s::StateAM, data::Data)
  f = Util.mixskewtlogpdf(s.mu', s.sigma', s.nu', s.phi', s.etaC',
                          data.yC_finite, dims=2)
  return data.ZC * log(s.gammaC) + data.NC_finite * log1p(-s.gammaC) + sum(f)
end

function loglike_T(s::StateAM, data::Data, beta::Bool)
  f = Util.skewtlogpdf.(s.mu', s.sigma', s.nu', s.phi', data.yT_finite)

  if beta
    gammaTstar = s.gammaT
    etaTstar = s.etaT
  else
    gammaTstar = s.gammaC
    etaTstar = s.etaC
  end

  f = Util.mixskewtlogpdf(s.mu', s.sigma', s.nu', s.phi', etaTstar',
                          data.yT_finite, dims=2)

  ll_inf = data.ZT * log(gammaTstar)
  ll_finite = data.NT_finite * log1p(-gammaTstar) + sum(f)

  return ll_inf + ll_finite
end
