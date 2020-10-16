function update_psi!(state::State, data::Data, prior::Prior)
  m, s = params(prior.psi)
  vkernel = zero.(state.psi)
  mkernel = zero.(state.psi)

  # Update kernels
  for i in samplenames
    yfinite = ref_yfinite(data, i)
    lam = ref_lambda(state, i)
    zeta = ref_zeta(state, i)
    v = ref_v(state, i)
    for n in eachindex(v)
      k = lam[n]
      vkernel[k] += zeta[n]^2 * v[n] / state.omega[k]
      mkernel[k] += zeta[n] * (yfinite[n] - state.mu[k]) * v[n] / state.omega[k]
    end
  end

  vnew = 1 ./ (s^-2 .+ vkernel)
  mnew = vnew .* (m/s^2 .+ mkernel)

  state.psi .= randn(prior.K) .* sqrt.(vnew) + mnew
end
