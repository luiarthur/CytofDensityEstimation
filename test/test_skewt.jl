@testset "SkewT" begin
  # Test it compiles.
  st = CDE.Util.SkewT(randn()*3, rand(), rand(LogNormal(3.5, .5)), rand()*3)

  skew = -3
  scale = 2
 
  altskew = CDE.Util.toaltskew(scale, skew)
  altscale = CDE.Util.toaltscale(scale, skew)

  @test isapprox(CDE.Util.scalefromaltskewt(altscale, altskew), scale)
  @test isapprox(CDE.Util.skewfromaltskewt(altscale, altskew), skew)
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


st1 = CDE.Util.SkewT(randn(), rand(), rand(LogNormal(2.5, .5)), randn()*3)
st2 = CDE.Util.SkewT(randn(), rand(), rand(LogNormal(2.5, .5)), randn()*3)
mm = MixtureModel([st1, st2], [.3, .7])
@time y = rand(mm, Int(1e6));
yy = collect(range(-6, 6, length=10000));
histogram(y, color=:grey, normalize=true, label=nothing, linecolor=:grey)
plot!(yy, pdf.(mm, yy), add=true, lw=2, color=:blue, label=nothing)
density!(y, label="kde", ls=:dot, color=:red, lw=3)
xlims!(minimum(x), maximum(x))
=#
