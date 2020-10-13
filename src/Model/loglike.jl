# TODO: Check math again!

"""
Marginal loglikelihood for sample i. Marginalizes over lambda and uses
skew-t likelihood.
"""
function marginal_loglike(gamma::AbstractFloat, 
                          eta::AbstractVector{<:Real},
                          loc::AbstractVector{<:Real},
                          scale::AbstractVector{<:Real},
                          df::AbstractVector{<:Real},
                          skew::AbstractVector{<:Real}, N::Int,
                          yfinite::AbstractVector{<:Real})
  Nfinite = length(yfinite)
  Z = N - Nfinite
  K = length(eta)

  f = logsumexp(Util.skewtlogpdf.(loc', scale', df', skew', yfinite) .+ 
                log.(eta)', dims=2)
  return Z * log(gamma) + Nfinite * log1p(-gamma) + sum(f)
end


"""
Marginal loglikelihood for sample i. Marginalizes over lambda and uses
Normal likelihood using latent variable representation of skew-t.
"""
function marginal_loglike_latent_var(gamma::AbstractFloat,
                                     eta::AbstractVector{<:Real},
                                     mu::AbstractVector{<:Real},
                                     psi::AbstractVector{<:Real},
                                     omega::AbstractVector{<:Real},
                                     nu::AbstractVector{<:Real},
                                     zeta::AbstractVector{<:Real},
                                     v::AbstractVector{<:Real}, N::Int,
                                     yfinite::AbstractVector{<:Real})
  Nfinite = length(yfinite)
  Z = N - Nfinite
  K = length(eta)

  loc = mu' .+ psi' .* zeta
  scale = sqrt.(omega)' ./ sqrt.(v)
  g = logsumexp(normlogpdf.(loc, scale, yfinite) .+ 
                gammalogpdf.(nu' / 2, 2 ./ nu', v) .+
                log.(eta)', dims=2)
  return Z * log(gamma) + Nfinite * log1p(-gamma) + sum(g)
end



"""
Loglikelihood for sample i. Does not marginalize over lambda, and uses Normal
likelihood using latent variable representation of skew-t.
"""
function loglike_latent_var(gamma::AbstractFloat,
                            lambda::Vector{Int},
                            mu::AbstractVector{<:Real},
                            psi::AbstractVector{<:Real},
                            omega::AbstractVector{<:Real},
                            nu::AbstractVector{<:Real},
                            zeta::AbstractVector{<:Real},
                            v::AbstractVector{<:Real}, N::Int,
                            yfinite::AbstractVector{<:Real})
  Nfinite = length(yfinite)
  Z = N - Nfinite

  loc = mu[lambda] + psi[lambda] .* zeta
  scale = sqrt.(omega)[lambda] ./ sqrt.(v)
  F = sum(normlogpdf.(loc, scale, yfinite))
  return Z * log(gamma) + Nfinite * log1p(-gamma) + sum(F)
end


"""
Loglikelihood for sample i. Does not marginalize over lambda, and uses Normal
likelihood using latent variable representation of skew-t. Arguments 
are `state::State` and `data::Data` only.
"""
function loglike_latent_var(state::State, data::Data)
  gammaC = state.gammaC
  gammaT = state.beta ? state.gammaT : state.gammaC

  llC = loglike_latent_var(gammaC, state.lambdaC, state.mu, state.psi,
                           state.omega, state.nu, state.zetaC, state.vC,
                           data.NC, data.yC_finite)

  llT = loglike_latent_var(gammaT, state.lambdaT, state.mu, state.psi,
                           state.omega, state.nu, state.zetaT, state.vT,
                           data.NT, data.yT_finite)

  return llC + llT
end
