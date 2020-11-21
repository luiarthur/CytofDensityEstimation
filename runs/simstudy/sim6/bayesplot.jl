using Distributions
using StatsPlots


plotpost(x; kwargs...) = _plotpost(x, false; kwargs...)
plotpost!(x; kwargs...) = _plotpost(x, true; kwargs...)

function _plotpost(x, add; cialpha=1, a=0.05, ms=8, truth=nothing, kwargs...)
  qs = [quantile(x, q) for q in [a/2, 1 - a/2]]
  if add
    histogram!(x, linealpha=0, grid=false, normalize=true, label=nothing; kwargs...)
  else
    histogram(x, linealpha=0, grid=false, normalize=true, label=nothing; kwargs...)
  end
  if cialpha > 0
    scatter!(qs, zero.(qs), ms=ms, alpha=cialpha, msc=kwargs[:color],
             color=kwargs[:color], label=nothing)
    # annotate!(qs[1], zero(qs[1]), text("<", 30, kwargs[:color]))
    # annotate!(qs[2], zero(qs[2]), text(">", 30, kwargs[:color]))
  end
  truth == nothing || vline!([truth], color=kwargs[:color], ls=:dot, lw=3)
  vline!([mean(x)], color=kwargs[:color])
end

# x = randn(10000)
# plotpost(x, color=:red, alpha=0.3, truth=0)
# plotpost!(x .+ 2, color=:blue, alpha=0.3, truth=2)

# scatter(randn(100), randn(100), msc=:blue, ms=10, color=:blue, alpha=0.5)
