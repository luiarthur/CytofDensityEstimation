import Pkg; Pkg.activate(joinpath(@__DIR__, "../../../"))

using Turing
using Distributions
using CytofDensityEstimation
using StatsPlots
import LinearAlgebra
const CDE = CytofDensityEstimation
include("scenarios.jl")

@model M1(yC_finite, yT_finite, K) = begin
  etaC ~ Dirichlet(K, 1/K)
  etaT ~ Dirichlet(K, 1/K)

  mu ~ filldist(Normal(0, 3), K)
  sigma ~ filldist(LogNormal(0, 1), K)
  nu ~ filldist(LogNormal(0, 1), K)
  phi ~ filldist(Normal(0, 6), K)

  yC_finite .~ MixtureModel(CDE.Util.SkewT.(mu, sigma, nu, phi), etaC)
  yT_finite .~ MixtureModel(CDE.Util.SkewT.(mu, sigma, nu, phi), etaT)
end
numparams(K, beta) = beta > 0 ? 6K : 5K
m1params = [:etaC, :etaT, :mu, :sigma, :nu, :phi]
get_last_state(sym, chain) = group(chain[end], sym).value.data
get_last_state(chain) = [get_last_state(sym, chain) for sym in m1params]
get_last_state(chain1)
getparam(sym, chain) = group(chain, sym).value.data[:, :, 1]

simdata = scenarios(1)
yC_finite = filter(isfinite, simdata[:yC])
yT_finite = filter(isfinite, simdata[:yT])
K = 6
m1 = M1(yC_finite, yT_finite, K)

function getinit(; s=[.1, .01, .01], nmcmc=fill(100, 3))
  init = nothing
  chain = nothing
  for i in 1:length(s)
    mh = MH(LinearAlgebra.I(numparams(K, 1)) * s[i])
    if i == 1
      chain = sample(m1, mh, nmcmc[i], save_state=true)
    else
      chain = sample(m1, mh, nmcmc[i], save_state=true, init_theta=init)
    end
    init = get_last_state(chain)
  end
  index_lp = findall(names(chain) .== :lp)[1]

  mat = hcat([getparam(sym, chain) for sym in m1params]...)
  return init, cov(unique(mat, dims=1))
end

init, _ = getinit(s=[.1, .01, .001], nmcmc=[100, 1000, 1000])
mh = MH(LinearAlgebra.I(numparams(K, 1)) * 0.001)
chain1 = sample(m1, mh, 3000, init_theta=init);
plot(vec(get(chain1, :lp)[1].data), label=nothing)

plot(group(chain1, :mu).value.data[:, :, 1], label=nothing)
plot(group(chain1, :etaC).value.data[:, :, 1], label=nothing)
plot(group(chain1, :etaT).value.data[:, :, 1], label=nothing)

param = Dict((sym, getparam(sym, chain1)) for sym in m1params)

function postpred(param; ygrid)
  B = size(param[:mu], 1)
  comps(b) = CDE.Util.SkewT.(param[:mu][b, :], param[:sigma][b, :], 
                             param[:nu][b, :], param[:phi][b, :])
  etaC(b) = param[:etaC][b, :]
  etaT(b) = param[:etaT][b, :]

  dC = hcat([pdf.(MixtureModel(comps(b), etaC(b)), ygrid) for b in 1:B]...)
  dT = hcat([pdf.(MixtureModel(comps(b), etaT(b)), ygrid) for b in 1:B]...)

  return (dC=dC, dT=dT)
end

ygrid = range(-6, 6, length=100)
pp = postpred(param, ygrid=ygrid)

pplower = vec(CDE.MCMC.quantiles(pp[:dC], .025, dims=2))
ppupper = vec(CDE.MCMC.quantiles(pp[:dC], .975, dims=2))
plot(ygrid, pplower, fillrange=ppupper, alpha=.3, color=:blue, label="C", legend=:topleft)
plot!(ygrid, pdf.(simdata[:mmC], ygrid), ls=:dot, color=:black, label=nothing)

pplower = vec(CDE.MCMC.quantiles(pp[:dT], .025, dims=2))
ppupper = vec(CDE.MCMC.quantiles(pp[:dT], .975, dims=2))
plot!(ygrid, pplower, fillrange=ppupper, alpha=.3, color=:red, label="T")
plot!(ygrid, pdf.(simdata[:mmT], ygrid), ls=:dot, color=:black, label=nothing)
