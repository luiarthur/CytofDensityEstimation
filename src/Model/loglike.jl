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
  return Z * log(gamma) + Nfinite * log1p(-gamma) * sum(f)
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
                gammalogpdf.(nu'/2, nu'/2, v) .+
                log.(eta)', dims=2)
  return Z * log(gamma) + Nfinite * log1p(-gamma) * sum(g)
end
