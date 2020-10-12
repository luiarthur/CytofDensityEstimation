@testset "SkewT" begin
  # Test it compiles.
  st = CDE.Util.SkewT(randn()*3, rand(), rand(LogNormal(3.5, .5)), rand()*3)
end

#= Test via plots
using StatsPlots

st = CDE.Util.SkewT(randn(), rand(), rand(LogNormal(2.5, .5)), randn()*3)
@time x = rand(st, Int(1e6));
xx = collect(range(-6, 6, length=10000));
histogram(x, color=:grey, normalize=true, label=nothing, linecolor=:grey)
plot!(xx, pdf.(st, xx), add=true, lw=2, color=:blue, label=nothing)
density!(x, label="kde", ls=:dot, color=:red, lw=3)
xlims!(minimum(x), maximum(x))
=#
