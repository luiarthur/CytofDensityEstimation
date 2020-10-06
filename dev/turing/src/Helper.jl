module Helper
using SpecialFunctions
using Distributions
import Random: AbstractRNG

std_t_lcdf(x, df) = log(std_t_cdf(x, df))
function std_t_cdf(x, df)
  x_t = df / (x^2 + df) 
  neg_cdf = 0.5 * beta_inc(0.5 * df, 0.5, x_t)[1]
  return x < 0 ? neg_cdf : 1 - neg_cdf
end

struct SkewT{Tdf<:Real, Tloc<:Real, Tscale<:Real, Tskew<:Real} <: ContinuousUnivariateDistribution
  df::Tdf
  loc::Tloc
  scale::Tscale
  skew::Tskew
end

function skewt_logpdf(df, loc, scale, skew, x)
  z = (x - loc) / scale
  u = skew * z * sqrt((df + 1) / (df + z ^ 2))
  # kernel = logpdf(TDist(df), z) + std_t_lcdf(u, df + 1)
  kernel = logpdf(TDist(df), z) + logcdf(TDist(df + 1), u)
  return kernel + log(2 / scale)
end

function Distributions.logpdf(d::SkewT{Tdf, Tloc, Tscale, Tskew}, x::Real) where {Tdf<:Real,
                                                                                  Tloc<:Real,
                                                                                  Tscale<:Real,
                                                                                  Tskew<:Real}
  return skewt_logpdf(d.df, d.loc, d.scale, d.skew, x)
end

function Distributions.rand(rng::AbstractRNG,
                            d::SkewT{Tdf, Tloc, Tscale, Tskew}) where {Tdf<:Real,
                                                                       Tloc<:Real,
                                                                       Tscale<:Real,
                                                                       Tskew<:Real}
  w = rand(rng, Gamma(d.df/2, 2/d.df))
  z = rand(rng, truncated(Normal(0, sqrt(1/w)), 0, Inf))
  delta = d.skew / sqrt(1 + d.skew ^ 2)
  loc = d.loc + d.scale * z * delta
  scale = d.scale * sqrt(1 - delta ^ 2)

  return randn(rng) * scale + loc
end

end
