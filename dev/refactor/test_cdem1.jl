import Pkg; Pkg.activate("../../")
using StatsPlots
using CytofDensityEstimation; const cde = CytofDensityEstimation
using CytofDensityEstimation: Util.SkewT
using CytofDensityEstimation: Util.skewfromaltskewt, Util.scalefromaltskewt
import Random
include("cdem1.jl")

Random.seed!(1234);
yC = rand(SkewT(2, .7, 7, -10), 1000)
yT = rand(SkewT(-1, .7, 7, -3), 1000)
histogram(yC, label=nothing)
histogram(yT, label=nothing)
K = 3

Random.seed!(0);
m, spl = generate_model_and_sampler(yC, yT, K, eps=0.01, L=20);
exclude = [:vC, :vT, :zetaC, :zetaT, :lambdaC, :lambdaT]
@time chain = litesample(m, spl, 20, discard_initial=10, thinning=1,
                         buffersize=10, exclude=exclude);
# @time chain = litesample(m, spl, 2000, discard_initial=1000, thinning=1,
#                          buffersize=100, exclude=exclude);

lp = vec(get(chain, :lp)[1].data);
plot(lp, label=nothing)

boxplot(group(chain, :etaC).value.data[:, :, 1], label=nothing)
boxplot(group(chain, :etaT).value.data[:, :, 1], label=nothing)
boxplot(group(chain, :mu).value.data[:, :, 1], label=nothing)
boxplot(group(chain, :omega).value.data[:, :, 1], label=nothing)
boxplot(group(chain, :nu).value.data[:, :, 1], label=nothing)
boxplot(group(chain, :psi).value.data[:, :, 1], label=nothing)

omega = group(chain, :omega).value.data[:, :, 1]
psi = group(chain, :psi).value.data[:, :, 1]
phi = skewfromaltskewt.(sqrt.(omega), psi)
sigma = scalefromaltskewt.(sqrt.(omega), psi)

plot(group(chain, :tau).value.data[:, 1], label=nothing)
plot(group(chain, :etaC).value.data[:, :, 1], label=nothing)
plot(group(chain, :etaT).value.data[:, :, 1], label=nothing)
plot(group(chain, :mu).value.data[:, :, 1], label=nothing)
plot(group(chain, :nu).value.data[:, :, 1], label=nothing)
plot(phi, label=nothing)
plot(sigma, label=nothing)


# DEMO
import Pkg; Pkg.activate("../../")
using StatsPlots
using CytofDensityEstimation: Util.SkewT
import Random
include("cdem1.jl")

@model function demo(y)
  nobs = length(y)
  nu ~ LogNormal(3, 1)
  omega ~ arraydist(Gamma.(fill(nu, nobs)/2, 2/nu))  # length nobs
  z ~ arraydist(truncated.(Normal.(0, 1 ./ sqrt.(omega)), 0, Inf))  # length nobs
  mu ~ Normal(0, 3)
  tau ~ Gamma(0.5, 1)
  sigmasq ~ InverseGamma(2.5, tau)
  psi ~ Normal(0, 3)
  y .~ Normal.(mu .+ psi * z, sqrt.(sigmasq ./ omega))
end

# Generate data
Random.seed!(1234);
y = rand(SkewT(3, .7, 7, -10), 500)
histogram(y)

# Fit model
spl = HMC(0.01, 1)
chain = sample(demo(y), spl, 200);
litechain = litesample(demo(y), spl, 200, discard_initial=50, thinning=1,
                       buffersize=50, exclude=[:omega, :z]);


