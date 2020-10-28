function update_beta_via_pseudo_prior!(state::State,
                                       state0::State, state1::State, 
                                       data::Data, prior::Prior)
  # Log priors
  lp0 = marginal_logprior(state0, prior)
  lp1 = marginal_logprior(state1, prior)

  # Log likelihoods
  ll0 = marginal_loglike(data, Dict(:mu=>state0.mu, :omega=>state0.omega,
                                    :psi=>state0.psi, :nu=>state0.nu,
                                    :beta=>state0.beta,
                                    :etaC=>state0.etaC, :gammaC=>state0.gammaC,
                                    :etaT=>state0.etaT, :gammaT=>state0.gammaT))
  ll1 = marginal_loglike(data, Dict(:mu=>state1.mu, :omega=>state1.omega,
                                    :psi=>state1.psi, :nu=>state1.nu,
                                    :beta=>state1.beta,
                                    :etaC=>state1.etaC, :gammaC=>state1.gammaC,
                                    :etaT=>state1.etaT, :gammaT=>state1.gammaT))

  # Log acceptance ratio
  log_acceptance_ratio_1 = (lp1 + ll1) - (lp0 + ll0) + logit(state.p)
  log_acceptance_ratio = state.beta ? -log_acceptance_ratio_1 : log_acceptance_ratio_1
  # println("(beta, lp0 - lp1): ($(state.beta), $(lp0 - lp1))")
  # println("(beta, ll0 - ll1): ($(state.beta), $(ll0 - ll1))")

  accept = log_acceptance_ratio > log(rand())
  state.beta = accept ? !state.beta : state.beta
end
