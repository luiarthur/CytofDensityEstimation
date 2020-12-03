using Turing
using Distributions

make_priors(K) = (a_tau=0.5, b_tau=1, a_omega=2.5, m_mu=0, s_mu=3,
                  m_psi=-1, s_psi=0.5, m_nu=2, s_nu=0.5, a_eta=fill(1/K, K))

function make_aux(y)
  v = one.(y)
  zeta = rand.(truncated.(Normal.(0, 1 ./ sqrt.(v)), 0, Inf))
  return (v=v, zeta=zeta)
end

@model function CDEM1(yC, yT, vC, zetaC, vT, zetaT, K; priors=make_priors(K))
  NC = length(yC)
  NT = length(yT)

  etaC ~ Dirichlet(K, 1/K)
  etaT ~ Dirichlet(K, 1/K)
  lambdaC ~ filldist(Categorical(etaC), NC)
  lambdaT ~ filldist(Categorical(etaT), NT)

  mu ~ filldist(Normal(priors[:m_mu], priors[:s_mu]), K)
  psi ~ filldist(Normal(priors[:m_psi], priors[:s_psi]), K)
  tau ~ Gamma(priors[:a_tau], 1/priors[:b_tau])
  omega ~ arraydist(InverseGamma.(fill(priors[:a_omega], K), tau))
  nu ~ filldist(LogNormal(priors[:m_nu], priors[:s_nu]), K)

  for n in 1:NC
    k = lambdaC[n]
    yC[n] ~ Normal(mu[k] + psi[k] * zeta[n], sqrt(omega[k] ./ vC[n]))
  end

  for n in 1:NT
    k = lambdaT[n]
    yT[n] ~ Normal(mu[k] + psi[k] * zeta[n], sqrt(omega[k] ./ vT[n]))
  end
end

function update_v!(c, y, lambda, v, zeta)
  nu = c.nu[lambda]
  omega = c.omega[lambda]
  psi = c.psi[lambda]
  mu = c.mu[lambda]

  shape = nu/2 .+ 1
  rate = (nu + zeta .^ 2 + ((y - mu - psi .* zeta) .^ 2) ./ omega) / 2

  for n in eachindex(v)
    v[n] = rand(Gamma(shape[n], 1/rate[n]))
  end
end

function update_zeta!(c, y, lambda, v, zeta)
  psi = c.psi[lambda]
  omega = c.omega[lambda]
  mu = c.mu[lambda]

  vnew = 1 ./ (v + (psi .^ 2) .* v ./ omega)
  mnew = vnew .* v .* psi .* (y - mu) ./ omega

  for n in eachindex(zeta)
    zeta[n] = rand(truncated(Normal(mnew[n], sqrt(vnew[n])), 0, Inf))
  end
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

function make_cond(yC, yT, vC, vT, zetaC, zetaT, K; priors=make_priors(K), skew=true, tdist=true)
  NC = length(yC)
  NT = length(yT)

  cond_lambdaC(c) = cond_lambda(c, c.etaC, yC, vC, zetaC, tdist)
  cond_lambdaT(c) = cond_lambda(c, c.etaT, yT, vT, zetaT, tdist)
  update_zetaC!(c) = update_zeta!(c, yC, c.lambdaC, vC, zetaC)
  update_zetaT!(c) = update_zeta!(c, yT, c.lambdaT, vT, zetaT)
  update_vC!(c) = update_v!(c, yC, c.lambdaC, vC, zetaC)
  update_vT!(c) = update_v!(c, yT, c.lambdaT, vT, zetaT)

  function update_vzeta!(c)
    update_zetaC!(c)
    update_zetaT!(c)
    update_vC!(c)
    update_vT!(c)
  end

  return (lambdaC=cond_lambdaC, lambdaT=cond_lambdaT, vzeta=update_vzeta!)
end

function generate_model_and_sampler(yC, yT, K; priors=make_priors(K))
  vC, zetaC = make_aux(yC)
  vT, zetaT = make_aux(yT)

  m = CDEM1(yC, yT, vC, zetaC, vT, zetaT, K, priors=priors)
  spl = let
    cond = make_cond(yC, yT, vC, vT, zetaC, zetaT, K, priors=priors)
  end

  function cond_lambdaC(c)
    cond[:update_vzeta](c)
    return cond[:lambdaC]
  end

  return m, Gibbs(GibbsConditional(:lambdaC, cond_lambdaC),
                  GibbsConditional(:lambdaT, cond[:lambdaT]))
end
