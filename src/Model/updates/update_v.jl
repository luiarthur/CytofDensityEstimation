function update_v!(state::State, data::Data, prior::Prior)
  for i in samplenames
    update_v!(i, state, data, prior)
  end
end


function update_v!(i::Char, state::State, data::Data, prior::Prior)
  lami = ref_lambda(state, i)
  zetai = ref_lambda(state, i)
  yi = ref_yfinite(data, i)
  vi = ref_v(state, i)

  nu = state.nu[lami]
  omega = state.omega[lami]
  psi = state.psi[lami]
  mu = state.mu[lami]

  shape = nu/2 .+ 1
  rate = (nu + zetai.^2 + ((yi + mu - psi .* zetai) .^ 2) ./ omega) / 2

  for n in eachindex(vi)
    vi[n] = rand(Gamma(shape[n], 1/rate[n]))
  end
end
