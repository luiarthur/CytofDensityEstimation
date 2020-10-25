# TODO: Check math again!

"""
Marginal loglikelihood for sample i. Marginalizes over lambda and uses skew-t
likelihood. Used for updating beta.
"""
function marginal_loglike_beta(i::Char, state::State, data::Data)
  Nfinite = ref_Nfinite(data, 'T')
  Z = ref_Z(data, 'T')
  yfinite = ref_yfinite(data, 'T')

  loc = state.mu
  scale = Util.scalefromaltskewt.(sqrt.(state.omega), state.psi)
  skew = Util.skewfromaltskewt.(sqrt.(state.omega), state.psi)
  df = state.nu

  eta = ref_eta(state, i)
  gamma = ref_gamma(state, i)

  f = logsumexp(Util.skewtlogpdf.(loc', scale', df', skew', yfinite) .+ 
                log.(eta)', dims=2)

  return Z * log(gamma) + Nfinite * log1p(-gamma) + sum(f)
end


function marginal_loglike(state::State, data::Data)
  llC = marginal_loglike('C', state, data)
  llT = marginal_loglike('T', state, data)
  return log(state.p) * llC + log1p(-state.p) * llT
end


"""
Marginal loglikelihood for sample i. Marginalizes over lambda and uses Normal
likelihood using latent variable representation of skew-t.  Used for updating
beta.
"""
function marginal_loglike_beta_latent_var(i::Char, state::State, data::Data)
  Nfinite = ref_Nfinite(data, 'T')
  yfinite = ref_yfinite(data, 'T')
  Z = ref_Z(data, 'T')

  K = length(state.mu)
  zeta = ref_zeta(state, 'T')
  v = ref_v(state, 'T')

  eta = ref_eta(state, i)
  gamma = ref_gamma(state, i)

  loc = state.mu' .+ state.psi' .* zeta
  scale = sqrt.(state.omega)' ./ sqrt.(v)
  g = logsumexp(normlogpdf.(loc, scale, yfinite) .+ 
                gammalogpdf.(state.nu' / 2, 2 ./ state.nu', v) .+
                log.(eta)', dims=2)

  return Z * log(gamma) + Nfinite * log1p(-gamma) + sum(g)
end


"""
Loglikelihood for sample i. Does not marginalize over lambda, and uses Normal
likelihood using latent variable representation of skew-t.
"""
function loglike_latent_var(i::Char, state::State, data::Data, use_star::Bool)
  Nfinite = ref_Nfinite(data, i)
  Z = ref_Z(data, i)
  yfinite = ref_yfinite(data, i)

  zeta = ref_zeta(state, i)
  v = ref_v(state, i)
  lambda = ref_lambda(state, i)

  j = (use_star && (i == 'T')) ? (state.beta ? 'T' : 'C') : i
  gamma = ref_gamma(state, j)

  loc = state.mu[lambda] + state.psi[lambda] .* zeta
  scale = sqrt.(state.omega)[lambda] ./ sqrt.(v)
  F = sum(normlogpdf.(loc, scale, yfinite))
  return Z * log(gamma) + Nfinite * log1p(-gamma) + sum(F)
end


"""
Loglikelihood for sample i. Does not marginalize over lambda, and uses Normal
likelihood using latent variable representation of skew-t. Arguments 
are `state::State` and `data::Data` only.
"""
function loglike_latent_var(state::State, data::Data)
  llC = loglike_latent_var('C', state, data, false)
  llT = loglike_latent_var('T', state, data, true)
  return llC + llT
end


### TODO: Bayes Factors (Check all below here.) ###

# TODO: Marginal loglikelihood for computing Bayes factor
# and posterior odds. Check!
function marginal_loglike(data::Data, state::Dict{Symbol, Any})
  llC = marginal_loglike_C(data, state)
  llT = marginal_loglike_T(data, state)
  return llC + llT
end


# TODO: Check.
function marginal_loglike_C(data::Data, state::Dict{Symbol, Any})
  loc = state[:mu]
  scale = Util.scalefromaltskewt.(sqrt.(state[:omega]), state[:psi])
  skew = Util.skewfromaltskewt.(sqrt.(state[:omega]), state[:psi])
  df = state[:nu]

  etaC = state[:etaC]
  gammaC = state[:gammaC]

  f = logsumexp(Util.skewtlogpdf.(loc', scale', df', skew', data.yC_finite) .+ 
                log.(etaC)', dims=2)

  return data.ZC * log(gammaC) + data.NC_finite * log1p(-gammaC) + sum(f)
end


# TODO: Check.
function marginal_loglike_T(data::Data, state::Dict{Symbol, Any})
  loc = state[:mu]
  scale = Util.scalefromaltskewt.(sqrt.(state[:omega]), state[:psi])
  skew = Util.skewfromaltskewt.(sqrt.(state[:omega]), state[:psi])
  df = state[:nu]

  etaT_star = state[:beta] ? state[:etaT] : state[:etaC]
  gammaT_star = state[:beta] ? state[:gammaT] : state[:gammaC]

  f = logsumexp(Util.skewtlogpdf.(loc', scale', df', skew', data.yT_finite) .+ 
                log.(etaT_star)', dims=2)

  return data.ZT * log(gammaT_star) + data.NT_finite * log1p(-gammaT_star) + sum(f)
end


"""
Bayes factor in favor of model where β=1.
See: https://projecteuclid.org/download/pdfview_1/euclid.ba/1346158782
"""
function bayes_factor(data::Data,
                      chain0::Vector{Dict{Symbol, Any}},
                      chain1::Vector{Dict{Symbol, Any}})
  ll0 = [marginal_loglike(data, state) for state in chain0]
  ll1 = [marginal_loglike(data, state) for state in chain1]
  B0 = length(ll0)
  B1 = length(ll1)
  ll0_mean = logsumexp(-ll0) - log(B0)
  ll1_mean = logsumexp(-ll1) - log(B1)

  # Ratio of harmonic means.
  return ll0_mean - ll1_mean
end


# TODO: Check.
function posterior_prob1(data::Data,
                         chain0::Vector{Dict{Symbol, Any}},
                         chain1::Vector{Dict{Symbol, Any}};
                         logpriorodds::Real=0)
  bf = bayes_factor(data, chain0, chain1)
  return logistic(bf + logpriorodds)
end


function posterior_prob1(bf::Real; logpriorodds::Real=0)
  return logistic(bf + logpriorodds)
end
