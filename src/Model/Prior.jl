struct Prior{F <: AbstractFloat}
  K::Int
  p::Beta
  gamma::Beta
  eta::Dirichlet
  mu::Normal
  a_omega::F
  nu::LogNormal
  psi::Normal
  tau::Gamma
end

function compute_prior_mu(data::Data)
  yfinite = [data.yC_finite; data.yT_finite]
  return Normal(mean(yfinite), std(yfinite))
end

function Prior(K; p=Beta(100, 100), gamma=Beta(1,1), eta=nothing,
               mu=Normal(0, 3), a_omega=2.5, tau=Gamma(0.5, 1),
               nu=LogNormal(0.5, 0.1), # nu=LogNormal(1.6, 0.4),
               psi=Normal(-1, 1), data=nothing)
  eta == nothing && (eta = Dirichlet(K, 1/K))
  data == nothing || (mu = compute_prior_mu(data))
  return Prior(K, p, gamma, eta, mu, a_omega, nu, psi, tau)
end

#= TEST:
prior = Prior(3)
for fn in fieldnames(Prior)
  println("$(fn) => $(getfield(prior, fn))")
end
=#
