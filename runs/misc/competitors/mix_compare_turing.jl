import Pkg; Pkg.activate("../../../")

using ProgressBars
using Turing
using Distributions
using BSON
using StatsPlots
using CytofDensityEstimation; const cde = CytofDensityEstimation
using StatsFuns
import CytofDensityEstimation: Util.SkewT
import Random
import LinearAlgebra
eye(n) = LinearAlgebra.I(n) * 1.0

function mixlpdf(loc, scale, nu, skew, y, w) 
  lm = logsumexp(cde.Util.skewtlogpdf.(loc', scale', nu', skew', y) .+
                 log.(w)', dims=2)
  return sum(lm)
end


# mixnormal
@model mixnormal(y, K) = begin
  mu ~ filldist(Normal(0, 1), K)
  sigmasq ~ filldist(InverseGamma(2, 3), K)
  sigma = sqrt.(sigmasq)
  eta ~ Dirichlet(K, 1/K)
  y .~ UnivariateGMM(mu, sigma, Categorical(eta))
end

# mixskewnormal
@model mixskewnormal(y, K) = begin
  mu ~ filldist(Normal(0, 1), K)
  sigmasq ~ filldist(InverseGamma(2, 3), K)
  phi ~ filldist(Normal(0, 3), K)
  sigma = sqrt.(sigmasq)
  eta ~ Dirichlet(K, 1/K)
  # y .~ MixtureModel(SkewT.(mu, sigma, 10000, phi), eta)
  Turing.acclogp!(_varinfo, mixlpdf(mu, sigma, 10000, phi, y, eta))
end

# mixskewt
@model mixskewt(y, K) = begin
  mu ~ filldist(Normal(0, 1), K)
  sigmasq ~ filldist(InverseGamma(2, 3), K)
  phi ~ filldist(Normal(0, 3), K)
  nu ~ filldist(LogNormal(3, 1), K)
  sigma = sqrt.(sigmasq)
  eta ~ Dirichlet(K, 1/K)
  # y .~ MixtureModel(SkewT.(mu, sigma, nu, phi), eta)
  Turing.acclogp!(_varinfo, mixlpdf(mu, sigma, nu, phi, y, eta))
end

function fitmodel(model, y, K, burn=10000, thin=10, nsamps=10000, eps=0.001)
  spl = if model == mixnormal
    MH(eye(3K) * eps)
  elseif model == mixskewnormal
    MH(eye(4K) * eps)
  elseif model == mixskewt
    MH(eye(5K) * eps)
  end

  m = model(y, K)
  return m, sample(m, spl, burn + nsamps)[burn:thin:end]
end

# Generate data.
Random.seed!(0);
y = rand(cde.Util.SkewT(2, 1, 10, -10), 500);

# Plot data.
histogram(y, bins=70, label=nothing, la=0)

# Fit models
K = 5
mixn, mixnormalchain = fitmodel(mixnormal, y, K);
mixsn, mixskewnormalchain = fitmodel(mixskewnormal, y, K);
mixst, mixskewtchain = fitmodel(mixskewt, y, K);

# Plots
plot(vec(get(mixnormalchain, :lp)[1].data))
plot(vec(get(mixskewnormalchain, :lp)[1].data))
plot(vec(get(mixskewtchain, :lp)[1].data))

plot(group(mixnormalchain, :mu))
plot(group(mixnormalchain, :sigmasq))
plot(group(mixnormalchain, :eta))

plot(group(mixskewnormalchain, :mu))
plot(group(mixskewnormalchain, :sigmasq))
plot(group(mixskewnormalchain, :phi))
plot(group(mixskewnormalchain, :eta))

plot(group(mixskewtchain, :mu))
plot(group(mixskewtchain, :sigmasq))
plot(group(mixskewtchain, :nu))
plot(group(mixskewtchain, :phi))
plot(group(mixskewtchain, :eta))

# mh = MH(LinearAlgebra.I(5K)/100)
