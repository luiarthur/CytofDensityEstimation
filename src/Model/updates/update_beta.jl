function update_beta!(state::State, data::Data, prior::Prior, tuners::Tuners)
  scale_skew = Matrix(hcat(Util.fromaltskewt.(sqrt.(state.omega), state.psi)...))
  scale = scale_skew[1, :]
  skew = scale_skew[2, :]

  llC = marginal_loglike(state.gammaC, state.etaC,
                         state.mu, scale, state.nu, skew, data.NT, 
                         data.yT_finite)

  llT = marginal_loglike(state.gammaT, state.etaT,
                         state.mu, scale, state.nu, skew, data.NT, 
                         data.yT_finite)

  p1 = logistic(logit(state.p) + llT - llC)

  state.beta = p1 > rand()
end
