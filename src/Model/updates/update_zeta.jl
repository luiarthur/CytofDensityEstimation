function update_zeta!(state::State, data::Data, prior::Prior)
  for i in samplenames
    update_zeta!(i, state, data, prior)
  end
end


function update_zeta!(i::Char, state::State, data::Data, prior::Prior)
  yi = ref_yfinite(data, i)
  zetai = ref_zeta(state, i)
  lami = ref_lambda(state, i)
  vi = ref_v(state, i)

  psi = state.psi[lami]
  omega = state.omega[lami]
  mu = state.mu[lami]

  vnew = 1 ./ (vi + (psi .^ 2) .* vi ./ omega)
  mnew = vnew .* vi .* psi .* (yi - mu) ./ omega

  for n in eachindex(zetai)
    zetai[n] = rand(truncated(Normal(mnew[n], sqrt(vnew[n])), 0, Inf))
  end
end
