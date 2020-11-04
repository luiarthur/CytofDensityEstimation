function logprior(state::StateAM, prior::PriorAM)
  lp = logpdf(prior.gamma, state.gammaC)
  lp += logpdf(prior.gamma, state.gammaT)
  lp += logpdf(prior.eta, state.etaC)
  lp += logpdf(prior.eta, state.etaT)
  lp += sum(logpdf.(prior.mu, state.mu))
  lp += sum(logpdf.(prior.sigma, state.sigma))
  lp += sum(logpdf.(prior.phi, state.phi))
  lp += sum(logpdf.(prior.nu, state.nu))
  return lp
end
