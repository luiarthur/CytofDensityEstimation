struct SkewT{Tloc<:Real, Tscale<:Real,
             Tdf<:Real, Tskew<:Real} <: ContinuousUnivariateDistribution
  loc::Tloc
  scale::Tscale
  df::Tdf
  skew::Tskew
end


function skewtlogpdf(loc::Real, scale::Real, df::Real, skew::Real, x::Real)
  z = (x - loc) / scale
  u = skew * z * sqrt((df + 1) / (df + z ^ 2))
  kernel = tdistlogpdf(df, z) + tdistlogcdf(df + 1, u)
  return kernel + logtwo - log(scale)
end


function skewtpdf(loc::Real, scale::Real, df::Real, skew::Real, x::Real)
  z = (x - loc) / scale
  u = skew * z * sqrt((df + 1) / (df + z ^ 2))
  kernel = tdistpdf(df, z) * tdistcdf(df + 1, u)
  return kernel * 2 / scale
end


function randskewt(rng::AbstractRNG, loc::Real, scale::Real, df::Real, skew::Real)
  w = rand(rng, Gamma(df/2, 2/df))
  inv_sqrt_w = sqrt(1 / w)
  z = rand(rng, truncated(Normal(0, inv_sqrt_w), 0, Inf))
  delta = skew / sqrt(1 + skew ^ 2)
  _loc = loc + scale * z * delta
  _scale = scale * sqrt(1 - delta ^ 2) * inv_sqrt_w

  return randn(rng) * _scale + _loc
end
function randskewt(loc::Real, scale::Real, df::Real, skew::Real)
  return randskewt(Random.GLOBAL_RNG, loc, scale, df, skew)
end


function Distributions.rand(rng::AbstractRNG,
                            d::SkewT{<:Real, <:Real, <:Real, <:Real})
  return randskewt(rng, d.loc, d.scale, d.df, d.skew)
end


function Distributions.logpdf(d::SkewT{<:Real, <:Real, <:Real, <:Real}, x::Real)
  return skewtlogpdf(d.loc, d.scale, d.df, d.skew, x)
end


function Distributions.pdf(d::SkewT{<:Real, <:Real, <:Real, <:Real}, x::Real)
  return skewtpdf(d.loc, d.scale, d.df, d.skew, x)
end


function SkewT(; loc, scale, altscale, altskew)
  scale, skew = fromaltskewt(altscale, altskew)
  SkewT(loc, scale, df, skew)
end


skewfromaltskewt(altscale, altskew) = altskew / sqrt(altscale)
scalefromaltskewt(altscale, altskew) = sqrt(altskew^2 + altscale^2)


function fromaltskewt(altscale, altskew)
  skew = skewfromaltskewt(altscale, altskew)
  scale = scalefromaltskewt(altscale, altskew)
  return [scale, skew]
end
