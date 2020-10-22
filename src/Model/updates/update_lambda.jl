function update_lambda!(state::State, data::Data, prior::Prior,
                        flags::Vector{Flag})
  update_lambdaC!(state, data, prior, flags)
  update_lambdaT!(state, data, prior, flags)
end


function sample_lambda(y::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                       psi::AbstractVector{<:Real},
                       omega::AbstractVector{<:Real},
                       nu::AbstractVector{<:Real},
                       eta::AbstractVector{<:Real},
                       zeta::AbstractVector{<:Real},
                       v::AbstractVector{<:Real}, flags::Vector{Flag})
  logeta = log.(eta)

  if UpdateLambdaWithSkewT() in flags
    st = [Util.AltSkewT(loc=mu[k], df=nu[k], altscale=sqrt(omega[k]),
                        altskew=psi[k]) for k in eachindex(eta)]
    lambda = [let
                logmix = logpdf.(st, y[n]) + logeta
                MCMC.wsample_logprob(logmix)
              end for n in eachindex(y)]
  else  # update with latent var representation.
    lambda = [let
                loc = mu + psi * zeta[n]
                scale = sqrt.(omega) / sqrt(v[n])
                logmix = normlogpdf.(loc, scale, y[n]) + 
                         gammalogpdf.(nu/2, 2 ./ nu, v[n]) + logeta
                MCMC.wsample_logprob(logmix)
              end for n in eachindex(y)]
  end
  return lambda
end


function update_lambdaC!(state::State, data::Data, prior::Prior,
                         flags::Vector{Flag})
  state.lambdaC .= sample_lambda(data.yC_finite,
                                 state.mu, state.psi, state.omega, state.nu,
                                 state.etaC, state.zetaC, state.vC, flags)
end


function update_lambdaT!(state::State, data::Data, prior::Prior,
                         flags::Vector{Flag})
  etaT_star = state.beta ? state.etaT : state.etaC
  state.lambdaT .= sample_lambda(data.yT_finite,
                                 state.mu, state.psi, state.omega, state.nu,
                                 etaT_star, state.zetaT, state.vT, flags)
end
