import Pkg; Pkg.activate("../")

using StatsPlots
using Turing
using Distributions
using StatsFuns
import Random

@model GMM(y, K) = begin
    mu ~ filldist(Normal(0, .3), K)
    sigmasq ~ filldist(InverseGamma(3, 2), K)
    sigma = sqrt.(sigmasq)
    eta ~ Dirichlet(K, 1/K)
    y .~ UnivariateGMM(mu, sigma, Categorical(eta))

    return (mu=mu, sigma=sigma, eta=eta)
end;

Random.seed!(1);
y = rand(UnivariateGMM([-2, 2], [0.5, 0.8], Categorical([.3, .7])), 1000)
histogram(y, bins=50)
m = GMM(y, 5)
chain = sample(m, NUTS(500, 0.65), 1000)
return_values = generated_quantities(m, chain)

mu = hcat(getfield.(return_values, :mu)...)
sigma = hcat(getfield.(return_values, :sigma)...)
eta = hcat(getfield.(return_values, :eta)...)

plot(mu')
mean(mu, dims=2)

yy = range(-6, 6, length=200)
ypred = [pdf.(UnivariateGMM(r[:mu], r[:sigma], Categorical(r[:eta])), yy)
         for r in return_values]
yp = hcat(ypred...)
plot(yy, mean(yp, dims=2))
histogram!(y, alpha=0.3, normalize=true, bins=50)
