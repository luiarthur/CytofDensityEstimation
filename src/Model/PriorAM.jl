struct PriorAM
  K::Int
  beta::Bool
  gamma::Beta
  eta::Dirichlet
  mu::Normal
  sigma::LogNormal
  nu::LogNormal
  phi::Normal
end


function PriorAM(K, beta; gamma=Beta(1,1), eta=nothing,
                 mu=Normal(0, 3), sigma=LogNormal(0, .5),
                 nu=LogNormal(1.6, 0.4), phi=Normal(-1, 10), data=nothing)
  eta == nothing && (eta = Dirichlet(K, 1/K))
  data == nothing || (mu = compute_prior_mu(data))
  return PriorAM(K, beta, gamma, eta, mu, sigma, nu, phi)
end


function logabsjacobian(v::AbstractVector{<:Real}, prior::PriorAM)
  K = prior.K

  out = MCMC.logabsjacobian_logitx(gammaCfromvec(K, v))
  out += MCMC.logabsjacobian_logitx(gammaTfromvec(K, v))

  out += MCMC.simplex_logabsdet(etaCfromvec(K, v))
  out += MCMC.simplex_logabsdet(etaTfromvec(K, v))
  out += sum(MCMC.logabsjacobian_logx.(sigmafromvec(K, v)))
  out += sum(MCMC.logabsjacobian_logx.(nufromvec(K, v)))

  return out
end
