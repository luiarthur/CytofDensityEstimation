function update_mu!(state::State, data::Data, prior::Prior)
  m_mu, s_mu = params(prior.mu)

  vkernels = zero.(state.mu)
  mkernels = zero.(state.mu)

  for i in samplenames
    yfinite = ref_yfinite(data, i)
    lambda = ref_lambda(state, i)
    zeta = ref_zeta(state, i)
    v = ref_v(state, i)
    for n in eachindex(lambda)
      k = lambda[n]
      vkernels[k] += v[n]
      mkernels[k] += (yfinite[n] - state.psi[k] * zeta[n]) * v[n]
    end
  end

  vnew = 1 ./ (s_mu^-2 .+ vkernels ./ state.omega)
  mnew = vnew .* (m_mu/(s_mu^2) .+ mkernels ./ state.omega)
  
  state.mu .= randn(prior.K) .* sqrt.(vnew) + mnew
end
