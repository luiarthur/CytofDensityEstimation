mutable struct State{F <: AbstractFloat}
  p::F
  beta::F
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
