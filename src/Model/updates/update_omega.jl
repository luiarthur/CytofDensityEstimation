function update_omega!(state::State, data::Data, prior::Prior)
  a, b = params(prior.omega)
  akernel = zero.(state.omega)
  bkernel = zero.(state.omega)

  for i in samplenames
    y = ref_yfinite(data, i)
    lam = ref_lambda(state, i)
    zeta = ref_zeta(state, i)
    v = ref_v(state, i)
    for n in eachindex(y)
      k = lam[n]
      akernel[k] += 1
      bkernel[k] += v[n] * (y[n] - state.mu[k] - state.psi[k] * zeta[n])^2
    end
  end

  anew = a .+ akernel / 2
  bnew = b .+ akernel / 2

  for k in 1:prior.K
    state.omega[k] = rand(InverseGamma(anew[k], bnew[k]))
  end
end
