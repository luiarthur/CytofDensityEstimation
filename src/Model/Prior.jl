struct Prior
  K::Int
  p::Beta
  gamma::Beta
  eta::Dirichlet
  mu::Normal
  omega::InverseGamma
  nu::LogNormal
  psi::Normal
end

function Prior(K; p=Beta(.5, .5), gamma=Beta(1,1), eta=nothing,
               mu=Normal(0, 3), omega=InverseGamma(3, 2),
               nu=LogNormal(3.5, 0.3), psi=Normal(-1, 3))
  eta != nothing || (eta = Dirichlet(K, 1/K))
  return Prior(K, p, gamma, eta, mu, omega, nu, psi)
end

#= TEST:
prior = Prior(3)
for fn in fieldnames(Prior)
  println("$(fn) => $(getfield(prior, fn))")
end
=#
