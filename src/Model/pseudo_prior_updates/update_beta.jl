# TODO

function update_beta_via_pseudo_prior!(state::State, data::Data, prior::Prior,
                                       flags::Vector{Flag})
  llC = marginal_loglike_beta('C', state, data)
  llT = marginal_loglike_beta('T', state, data)

  # p1 = logistic(logit(state.p) + llT - llC)
  # state.beta = p1 > rand()
end
