function marginal_loglike(gamma::AbstractFloat, 
                          eta::Vector{F}, loc::Vector{F},
                          scale::Vector{F}, df::Vector{F},
                          skew::Vector{F}, N::Int,
                          yfinite::Vector{G}) where {F <: AbstractFloat, G <: AbstractFloat}
  Nfinite = length(yfinite)
  Z = N - Nfinite
  K = length(eta)

  f = logsumexp(Util.skewtlogpdf.(loc', scale', df', skew', yfinite) .+ 
                log.(eta)', dims=2)
  return Z * log(gamma) + Nfinite * log1p(-gamma) * sum(f)
end
