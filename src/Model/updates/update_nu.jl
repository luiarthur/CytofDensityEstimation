function update_nu!(state::State, data::Data, prior::Prior, tuners::Tuners)
  # NOTE: This could be optimized.
  for k in eachindex(state.nu)
    update_nu!(k, state, data, prior, tuners)
  end
end


function update_nu!(k::Int, state::State, data::Data, prior::Prior,
                    tuners::Tuners)
  log_prob = function(log_nu_k)
    m, s = params(prior.nu)
    logprior = normlogpdf(m, s, log_nu_k)

    nu_k = exp(log_nu_k)
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

  state.nu[k] = exp(MCMC.metropolisAdaptive(log(state.nu[k]),
                                            log_prob, tuners.nu[k]))
end
