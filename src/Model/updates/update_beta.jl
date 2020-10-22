function update_beta!(state::State, data::Data, prior::Prior,
                      flags::Vector{Flag})
  if UpdateBetaWithSkewT() in flags
    llC = marginal_loglike_beta('C', state, data)
    llT = marginal_loglike_beta('T', state, data)
  else
    llC = marginal_loglike_beta_latent_var('C', state, data)
    llT = marginal_loglike_beta_latent_var('T', state, data)
  end

  p1 = logistic(logit(state.p) + llT - llC)
  state.beta = p1 > rand()
end
