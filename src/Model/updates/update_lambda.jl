function update_lambda!(state::State, data::Data, prior::Prior, tuners::Tuners)
  update_lambdaC!(state, data, prior)
  update_lambdaT!(state, data, prior)
end


function sample_lambda(y::Vector{Y}, mu::Vector{F}, psi::Vector{F},
                       omega::Vector{F}, eta::Vector{F},
                       zeta::Vector{F}, v::Vector{F}) where {Y<:AbstractFloat, F<:AbstractFloat}
  logeta = log.(eta)
  lambda = [let
              loc = mu + psi .* zeta[n]
              scale = sqrt.(omega) / sqrt(v[n])
              logmix = normlogpdf.(loc, scale, y[n]) .+ logeta
              MCMC.wsample_logprob(logmix)
            end for n in eachindex(y)]
  return lambda
end


function update_lambdaC!(state::State, data::Data, prior::Prior)
  state.lambdaC .= sample_lambda(data.yC_finite,
                                 state.mu, state.psi, state.omega, state.etaC,
                                 state.zetaC, state.vC)
end


function update_lambdaT!(state::State, data::Data, prior::Prior)
  etaT_star = state.beta ? state.etaT : state.etaC
  state.lambdaT .= sample_lambda(data.yT_finite,
                                 state.mu, state.psi, state.omega, etaT_star,
                                 state.zetaT, state.vT)
end
