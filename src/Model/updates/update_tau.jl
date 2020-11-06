function update_tau!(state::State, data::Data, prior::Prior)
  a, b = params(prior.tau)
  new_shape = a + prior.K * prior.a_omega
  new_rate = b + sum(1 ./ state.omega)
  state.p = rand(Gamma(new_shape, 1 / new_rate))
end
