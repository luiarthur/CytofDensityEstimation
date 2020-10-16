function update_nu!(state::State, data::Data, prior::Prior, tuners::Tuners)
  # NOTE: This could be optimized.
  for k in eachindex(state.nu)
    update_nu!(k, state, data, prior, tuners)
  end
end


function update_nu_old!(k::Int, state::State, data::Data, prior::Prior,
                    tuners::Tuners)
  log_prob = function(nu_k)
    logprior = logpdf(prior.nu, nu_k)

    loglike = zero(state.p)
    for i in samplenames
      lami = ref_lambda(state, i)
      vi = ref_v(state, i)
      for n in eachindex(vi)
        (lami[n] == k) && (loglike += gammalogpdf(nu_k/2, 2/nu_k, vi[n]))
      end
    end

    return logprior + loglike
  end

  state.nu[k] = MCMC.metLogAdaptive(state.nu[k], log_prob, tuners.nu[k])
end


realtodf(x) = exp(x) + 1
dftoreal(df) = log(df - 1)
mtod(m) = (m - 1) / (1 + sqrt(2))
function tdflogpdf(df, m)
  d = mtod(m)
  return (df >= 1) ? log(df - 1) - 3 * log(df - 1 + d) : -Inf
end
function tdfpdf(df, m)
  d = mtod(m)
  return (df >= 1) ? (df - 1) / (df -1 + d)^3 : 0
end


function update_nu!(k::Int, state::State, data::Data, prior::Prior,
                        tuners::Tuners)
  _logprob = function(nu_k)
    logprior = tdflogpdf(nu_k, 5)  # NOTE: Hardcoded m.

    loglike = zero(state.p)
    for i in samplenames
      lami = ref_lambda(state, i)
      vi = ref_v(state, i)
      for n in eachindex(vi)
        (lami[n] == k) && (loglike += gammalogpdf(nu_k/2, 2/nu_k, vi[n]))
      end
    end

    return logprior + loglike
  end

  logprob(x) = _logprob(realtodf(x)) + x
  x = dftoreal(state.nu[k])
  x = MCMC.metropolisAdaptive(x, logprob, tuners.nu[k])
  state.nu[k] = realtodf(x)
end
