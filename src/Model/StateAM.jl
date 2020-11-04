mutable struct StateAM{F <: AbstractFloat, V <: AbstractVector{<:Real}}
  gammaC::F
  gammaT::F
  etaC::V  # K
  etaT::V  # K
  mu::V  # K. mixture loc.
  sigma::V  # K. mixture scale.
  nu::V  # K. mixture df.
  phi::V  # K. mixture skew.
end


function StateAM(data::Data, prior::PriorAM)
  gammaC = rand(prior.gamma)
  gammaT = rand(prior.gamma)
  etaC = rand(prior.eta)
  etaT = rand(prior.eta)
  mu = rand(prior.mu, prior.K)
  sigma = rand(prior.sigma, prior.K)
  nu = rand(prior.nu, prior.K)
  phi = rand(prior.phi, prior.K)

  return StateAM(gammaC, gammaT, etaC, etaT, mu, sigma, nu, phi)
end


# TODO
function tovec(s::StateAM)
  return [s.mu; log.(s.sigma); log.(s.nu); s.phi;
          logit(s.gammaC); logit(s.gammaT);
          MCMC.simplex_transform(s.etaC);
          MCMC.simplex_transform(s.etaT)]
end


mufromvec(K::Integer, v::AbstractVector{<:Real}) = v[1:K]
sigmafromvec(K::Integer, v::AbstractVector{<:Real}) = v[K+1:2K]
nufromvec(K::Integer, v::AbstractVector{<:Real}) = v[2K+1:3K]
phifromvec(K::Integer, v::AbstractVector{<:Real}) = v[3K+1:4K]
gammaCfromvec(K::Integer, v::AbstractVector{<:Real}) = v[4K+1]
gammaTfromvec(K::Integer, v::AbstractVector{<:Real}) = v[4K+2]
etaCfromvec(K::Integer, v::AbstractVector{<:Real}) = v[4K+3:5K+1]
etaTfromvec(K::Integer, v::AbstractVector{<:Real}) = v[5K+2:6K]


function fromvec!(s::StateAM, v::AbstractVector{<:Real})
  K = length(s.mu)
  s.gammaC = logistic(gammaCfromvec(K, v))
  s.gammaT = logistic(gammaTfromvec(K, v))
  s.etaC = MCMC.simplex_invtransform(etaCfromvec(K, v))
  s.etaT = MCMC.simplex_invtransform(etaTfromvec(K, v))
  s.mu = mufromvec(K, v)
  s.sigma = exp.(sigmafromvec(K, v))
  s.nu = exp.(nufromvec(K, v))
  s.phi = phifromvec(K, v)
end
