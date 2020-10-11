mutable struct State{F <: AbstractFloat}
  p::F
  beta::Bool
  gammaC::F
  gammaT::F
  etaC::Vector{F}  # K
  etaT::Vector{F}  # K
  lambdaC::Vector{Int}  # NC
  lambdaT::Vector{Int}  # NT
  mu::Vector{F}  # K
  nu::Vector{F}  # K
  omega::Vector{F}  # K. alt variance.
  psi::Vector{F}  # K. alt skew.
  vC::Vector{F}  # NC
  vT::Vector{F}  # NT
  zetaC::Vector{F}  # NC
  zetaT::Vector{F}  # NT
end

function State(data::Data, prior::Prior)
  # TODO
  p = rand(prior.p)
  beta = true
  gammaC = rand(prior.gamma)
  gammaT = rand(prior.gamma)
  etaC = rand(prior.eta)
  etaT = rand(prior.eta)
  lambdaC = rand(Categorical(etaC), data.NC_finite)
  lambdaT = rand(Categorical(etaT), data.NT_finite)
  mu = rand(prior.mu, prior.K)
  nu = rand(prior.nu, prior.K)
  omega = rand(prior.omega, prior.K)
  psi = rand(prior.psi, prior.K)
  vC = rand(Gamma(mean(nu)/2, 2/mean(nu)), data.NC_finite)
  vT = rand(Gamma(mean(nu)/2, 2/mean(nu)), data.NT_finite)
  zetaC = [rand(truncated(Normal(0, sqrt(1/v)), 0, Inf)) for v in vC]
  zetaT = [rand(truncated(Normal(0, sqrt(1/v)), 0, Inf)) for v in vT]

  return State(p, beta, gammaC, gammaT, etaC, etaT, lambdaC, lambdaT,
               mu, nu, omega, psi, vC, vT, zetaC, zetaT)
end

