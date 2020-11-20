import Pkg; Pkg.activate(".")

using Distributions
using AdvancedMH
import Random
import LinearAlgebra

eye(n::Int) = Matrix(LinearAlgebra.I(n) * 1.0)

function update_mean!(current_mean, x, iter)
  current_mean .+= (x - current_mean) / iter
end

function update_cov!(current_cov, current_mean, x, iter)
  d = x - current_mean
  current_cov .= current_cov * (iter - 1)/iter + (d*d') * (iter - 1)/iter^2
  current_cov .= Matrix(LinearAlgebra.Symmetric(current_cov))
end

# FIXME: How to write this properly?
mutable struct RandomWalkAdaptiveProposal{P} <: AdvancedMH.Proposal{P}
  proposal::P
  iter::Int
end

d(p::RandomWalkAdaptiveProposal) = length(p.proposal.components[1].μ)

function RWAP(d::Int; beta=0.05, iter=1)
  p = MixtureModel([MvNormal(zeros(d), eye(d) * .01/d),
                    MvNormal(zeros(d), eye(d) * .01/d)],
                   [1-beta, beta])
  return RandomWalkAdaptiveProposal(p, iter)
end
function RWAP(d::Distribution; beta=0.05, iter=1)
  K = length(d)
  p = MixtureModel([d, MvNormal(zeros(K), eye(K) * .01/K)],
                   [1-beta, beta])
  return RandomWalkAdaptiveProposal(p, iter)
end


function update!(p::RandomWalkAdaptiveProposal, x)
  p.iter += 1
  _mean = p.proposal.components[1].μ
  _cov = Matrix(p.proposal.components[1].Σ)

  update_mean!(_mean, x, p.iter)
  update_cov!(_cov, _mean, x, p.iter)

  # Is there a better way?
  p.proposal.components[1] = MvNormal(_mean, _cov)

  return
end

function AdvancedMH.propose(rng::Random.AbstractRNG, p::RandomWalkAdaptiveProposal, m::DensityModel)
  return AdvancedMH.propose(rng, StaticProposal(p.proposal), m)
end
function AdvancedMH.propose(
  rng::Random.AbstractRNG,
  proposal::RandomWalkAdaptiveProposal{<:Union{Distribution,AbstractArray}}, 
  model::DensityModel, 
  t
)
  update!(proposal, t)
  if proposal.iter < 2 * d(proposal)
    out = rand(rng, MvNormal(t, proposal.proposal.components[2].Σ))
  else
    _cov = proposal.proposal.components[1].Σ * 2.38^2 / d(proposal)
    out = rand(rng, MvNormal(t, _cov))
  end
  return out
end
function AdvancedMH.q(
  proposal::RandomWalkAdaptiveProposal{<:Union{Distribution,AbstractArray}}, 
  t,
  t_cond
)
  # return logpdf(proposal, t - t_cond)
  return 0  # because the proposal is symmetric.
end

#= TEST
p = RWAP(3)
update!(p, randn(3))
=#

RWAM(d::Int; beta=0.05, iter=1) = AdvancedMH.MetropolisHastings(RWAP(d, beta=beta, iter=iter))
function RWAM(d::Distribution; beta=0.05, iter=1)
  return AdvancedMH.MetropolisHastings(RWAP(d, beta=beta, iter=iter))
end
