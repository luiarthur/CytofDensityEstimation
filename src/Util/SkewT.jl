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


function randskewt(rng::AbstractRNG, loc::Real, scale::Real, df::Real, skew::Real)
  w = rand(rng, Gamma(df/2, 2/df))
  inv_sqrt_w = sqrt(1 / w)
  z = rand(rng, truncated(Normal(0, inv_sqrt_w), 0, Inf))
  delta = skew / sqrt(1 + skew ^ 2)
  _loc = loc + scale * z * delta
  _scale = scale * sqrt(1 - delta ^ 2) * inv_sqrt_w

  return randn(rng) * _scale + _loc
end
randskewt(loc::Real, scale::Real, df::Real, skew::Real) = randskewt(Random.GLOBAL_RNG)


function Distributions.rand(rng::AbstractRNG,
                            d::SkewT{Tloc, Tscale, Tdf, Tskew}
                           ) where {Tloc <: Real, Tscale <: Real, Tdf <: Real, Tskew <: Real}
  return randskewt(rng, d.loc, d.scale, d.df, d.skew)
end


function Distributions.logpdf(d::SkewT{Tloc, Tscale, Tdf, Tskew}, x::Real) where {Tloc<:Real,
                                                                                  Tscale<:Real,
                                                                                  Tdf<:Real,
                                                                                  Tskew<:Real}
  return skewtlogpdf(d.loc, d.scale, d.df, d.skew, x)
end

function SkewT(; loc, scale, altscale, altskew)
  skew = altskew / sqrt(altscale)
  scale = sqrt(altskew^2 + altscale^2)
  SkewT(loc, scale, df, skew)
end


### TEST ###
#=
import Pkg; Pkg.activate("../../")
using Distributions
import Random: AbstractRNG, GLOBAL_RNG
using StatsFuns
using Plots
import StatsPlots: density

st = SkewT(randn()*3, rand(), rand(LogNormal(3.5, .5)), rand()*3)
@time x = rand(st, Int(1e6));
xx = collect(range(-6, 6, length=10000))
histogram(x, color=:grey, normalize=true, label=nothing, linecolor=:grey)
plot!(xx, pdf.(st, xx), add=true, lw=2, color=:blue, label=nothing)
=#
