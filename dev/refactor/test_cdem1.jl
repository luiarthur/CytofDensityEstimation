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
K = 2

Random.seed!(1234);
m, spl = generate_model_and_sampler(yC, yT, K, eps=0.01, L=20);
@time chain = sample(m, spl, 1000, discard_initial=1000);

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

plot(group(chain, :etaC).value.data[:, :, 1], label=nothing)
plot(group(chain, :etaT).value.data[:, :, 1], label=nothing)
plot(group(chain, :mu).value.data[:, :, 1], label=nothing)
plot(group(chain, :nu).value.data[:, :, 1], label=nothing)
plot(phi, label=nothing)
plot(sigma, label=nothing)
