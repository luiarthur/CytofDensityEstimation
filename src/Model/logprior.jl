# TODO: Check.
function logprior(state::State, prior::Prior)
  lp = sum(logpdf.(prior.mu, state.mu)) 
  lp += sum(logpdf.(prior.omega, state.omega)) 
  lp += sum(logpdf.(prior.psi, state.psi)) 
  lp += sum(logpdf.(prior.nu, state.nu)) 
  lp += logpdf(prior.gamma, state.gammaC)
  lp += logpdf(prior.gamma, state.gammaT)
  lp += logpdf(prior.eta, state.etaC)
  lp += logpdf(prior.eta, state.etaT)

  # lambda prior
  etaT_star = state.beta ? state.etaT : state.etaC
  lp += sum(log.(state.etaC[state.lambdaC]))
  lp += sum(log.(etaT_star[state.lambdaT]))

  # v prior
  nuC = state.nu[state.lambdaC]
  nuT = state.nu[state.lambdaT]
  lp += sum(gammalogpdf.(nuC/2, 2 ./ nuC, state.vC))
  lp += sum(gammalogpdf.(nuT/2, 2 ./ nuT, state.vT))

  # zeta prior
  lp += sum(logpdf(truncated(Normal(0, 1/sqrt(state.vC[n])), 0, Inf), state.zetaC[n])
            for n in eachindex(state.zetaC))
  lp += sum(logpdf(truncated(Normal(0, 1/sqrt(state.vT[n])), 0, Inf), state.zetaT[n])
            for n in eachindex(state.zetaT))

  return lp
end


function marginal_logprior(state::State, prior::Prior)
  lp = sum(logpdf.(prior.mu, state.mu)) 
  lp += sum(logpdf.(prior.omega, state.omega)) 
  lp += sum(logpdf.(prior.psi, state.psi)) 
  lp += sum(logpdf.(prior.nu, state.nu)) 
  lp += logpdf(prior.gamma, state.gammaC)
  lp += logpdf(prior.eta, state.etaC)
  if state.beta
    lp += logpdf(prior.gamma, state.gammaT)
    lp += logpdf(prior.eta, state.etaT)
  end
  return lp
end
