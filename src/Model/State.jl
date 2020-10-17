mutable struct State{F <: AbstractFloat, V <: AbstractVector{<:Real}}
  p::F
  beta::Bool
  gammaC::F
  gammaT::F
  etaC::V  # K
  etaT::V  # K
  lambdaC::Vector{Int}  # NC
  lambdaT::Vector{Int}  # NT
  mu::V  # K
  nu::V  # K
  omega::V  # K. alt variance.
  psi::V  # K. alt skew.
  vC::V  # NC
  vT::V  # NT
  zetaC::V  # NC
  zetaT::V  # NT
end


function State(data::Data, prior::Prior)
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
  vC = [rand(Gamma(nu[k]/2, 2/nu[k])) for k in lambdaC]
  vT = [rand(Gamma(nu[k]/2, 2/nu[k])) for k in lambdaT]
  zetaC = [rand(truncated(Normal(0, sqrt(1/v)), 0, Inf)) for v in vC]
  zetaT = [rand(truncated(Normal(0, sqrt(1/v)), 0, Inf)) for v in vT]

  return State(p, beta, gammaC, gammaT, etaC, etaT, lambdaC, lambdaT,
               mu, nu, omega, psi, vC, vT, zetaC, zetaT)
end


ref_lambda(state::State, i::Char) = (i == 'C') ? state.lambdaC : state.lambdaT
ref_v(state::State, i::Char) = (i == 'C') ? state.vC : state.vT
ref_zeta(state::State, i::Char) = (i == 'C') ? state.zetaC : state.zetaT

ref_eta(state::State, i::Char) = (i == 'C') ? state.etaC : state.etaT
ref_gamma(state::State, i::Char) = (i == 'C') ? state.gammaC : state.gammaT
