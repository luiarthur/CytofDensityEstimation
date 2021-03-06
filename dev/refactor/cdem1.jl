using Turing
using Distributions
using StatsFuns
import CytofDensityEstimation: MCMC
using ProgressBars
include("litesample.jl")

make_priors(K) = (a_tau=0.5, b_tau=1, a_omega=2.5, m_mu=0, s_mu=3,
                  m_psi=-1, s_psi=0.5, m_nu=2, s_nu=0.5, a_eta=fill(1/K, K))

@model function CDEM1(yC, yT, K; priors=make_priors(K))
  NC = length(yC)
  NT = length(yT)

  etaC ~ Dirichlet(priors[:a_eta])
  etaT ~ Dirichlet(priors[:a_eta])
  lambdaC ~ filldist(Categorical(etaC), NC)
  lambdaT ~ filldist(Categorical(etaT), NT)

  mu ~ filldist(Normal(priors[:m_mu], priors[:s_mu]), K)
  psi ~ filldist(Normal(priors[:m_psi], priors[:s_psi]), K)
  tau ~ Gamma(priors[:a_tau], 1/priors[:b_tau])
  omega ~ arraydist(InverseGamma.(fill(priors[:a_omega], K), tau))
  nu ~ filldist(LogNormal(priors[:m_nu], priors[:s_nu]), K)

  vC ~ arraydist(Gamma.(nu[lambdaC[1:end]] / 2, 2 ./ nu[lambdaC[1:end]]))
  vT ~ arraydist(Gamma.(nu[lambdaT[1:end]] / 2, 2 ./ nu[lambdaT[1:end]]))
  zetaC ~ arraydist(truncated.(Normal.(0, sqrt.(1 ./ vC)), 0, Inf))
  zetaT ~ arraydist(truncated.(Normal.(0, sqrt.(1 ./ vT)), 0, Inf))

  yC .~ Normal.(mu[lambdaC[1:end]] + psi[lambdaC[1:end]] .* zetaC[lambdaC[1:end]],
                sqrt.(omega[lambdaC[1:end]] ./ vC[lambdaC[1:end]]))
  yT .~ Normal.(mu[lambdaT[1:end]] + psi[lambdaT[1:end]] .* zetaT[lambdaT[1:end]],
                sqrt.(omega[lambdaT[1:end]] ./ vT[lambdaT[1:end]]))
end

function cond_v(c, y, lambda, zeta)
  nu = c.nu[lambda]
  omega = c.omega[lambda]
  psi = c.psi[lambda]
  mu = c.mu[lambda]

  shape = nu/2 .+ 1
  rate = (nu + zeta .^ 2 + ((y - mu - psi .* zeta) .^ 2) ./ omega) / 2

  return arraydist(Gamma.(shape, 1 ./ rate))
end

function cond_zeta(c, y, lambda, v)
  psi = c.psi[lambda]
  omega = c.omega[lambda]
  mu = c.mu[lambda]

  vnew = 1 ./ (v + (psi .^ 2) .* v ./ omega)
  mnew = vnew .* v .* psi .* (y - mu) ./ omega

  return arraydist(truncated.(Normal.(mnew, sqrt.(vnew)), 0, Inf))
end

function cond_lambda(c, y, eta, v, zeta, tdist)
  logeta = log.(eta)
  lambda_prob = [let
                   loc = c.mu + c.psi * zeta[n]
                   scale = sqrt.(c.omega) / sqrt(v[n])
                   logmix = normlogpdf.(loc, scale, y[n]) + logeta
                   tdist && (logmix .+= gammalogpdf.(c.nu / 2, 2 ./ c.nu, v[n]))
                   logmix .-= logsumexp(logmix)
                   exp.(logmix)
                 end for n in eachindex(y)]
  return arraydist(Categorical.(lambda_prob))
end

function cond_eta(c, lambda, priors)
  anew = priors[:a_eta] .+ 0
  for n in eachindex(lambda)
    anew[lambda[n]] += 1
  end
  return Dirichlet(anew)
end

function make_cond(yC, yT, K; priors=make_priors(K), skew=true, tdist=true)
  NC = length(yC)
  NT = length(yT)

  # TODO: Change behavior with skewt and tist
  cond_lambdaC(c) = cond_lambda(c, yC, c.etaC, c.vC, c.zetaC, tdist)
  cond_lambdaT(c) = cond_lambda(c, yT, c.etaT, c.vT, c.zetaT, tdist)
  cond_etaC(c) = cond_eta(c, c.lambdaC, priors)
  cond_etaT(c) = cond_eta(c, c.lambdaT, priors)
  cond_vC(c) = cond_v(c, yC, c.lambdaC, c.zetaC)
  cond_vT(c) = cond_v(c, yT, c.lambdaT, c.zetaT)
  cond_zetaC(c) = cond_zeta(c, yC, c.lambdaC, c.vC)
  cond_zetaT(c) = cond_zeta(c, yT, c.lambdaT, c.vT)

  return (lambdaC=cond_lambdaC, lambdaT=cond_lambdaT, 
          zetaC=cond_zetaC, zetaT=cond_zetaT,
          vC=cond_vC, vT=cond_vT,
          etaC=cond_etaC, etaT=cond_etaT)
end

make_hmc_sampler(eps, L) = HMC(eps, L, :mu, :omega, :nu, :psi, :tau)

function generate_model_and_sampler(yC, yT, K; priors=make_priors(K), sampler_other=nothing,
                                    eps=0.01, L=100)
  m = CDEM1(yC, yT, K, priors=priors)
  cond = make_cond(yC, yT, K, priors=priors)

  sampler_other == nothing && (sampler_other = make_hmc_sampler(eps, L))

  return m, Gibbs(GibbsConditional(:etaC, cond[:etaC]),
                  GibbsConditional(:etaT, cond[:etaT]),
                  GibbsConditional(:lambdaC, cond[:lambdaC]),
                  GibbsConditional(:lambdaT, cond[:lambdaT]),
                  GibbsConditional(:vC, cond[:vC]),
                  GibbsConditional(:vT, cond[:vT]),
                  GibbsConditional(:zetaC, cond[:zetaC]),
                  GibbsConditional(:zetaT, cond[:zetaT]),
                  GibbsConditional(:vC, cond[:vC]),
                  GibbsConditional(:vT, cond[:vT]),
                  GibbsConditional(:zetaC, cond[:zetaC]),
                  GibbsConditional(:zetaT, cond[:zetaT]),
                  sampler_other)
end
