function update_p!(state::State, data::Data, prior::Prior)
  a, b = params(prior.p)
  state.p = rand(Beta(a + state.beta, b + 1 - state.beta))
end
