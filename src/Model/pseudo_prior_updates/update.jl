# TODO: Check.

include("update_beta.jl")

function _update_state!(state::State, data::Data, prior::Prior,
                        tuners::Tuners; rep_aux::Integer=1, 
                        fix::Vector{Symbol}=Symbol[],
                        flags::Vector{Flag}=Flag[])
  update_state!(state, data, prior, tuners, fix=[fix; [:p, :beta]],
                flags=flags)
  for _ in 1:rep_aux
    update_state!(state, data, prior, tuners, flags=flags,
                  fix=[fix; [:p, :beta, :lambda, :gamma, :eta]])
  end
end

function update_state_via_pseudo_prior!(state::State,
                                        state0::State, state1::State,
                                        data::Data, prior::Prior,
                                        tuners::Tuners,
                                        tuners0::Tuners, tuners1::Tuners;
                                        rep_aux::Integer=1,
                                        fix::Vector{Symbol}=Symbol[],
                                        flags::Vector{Flag}=Flag[])
  # State: beta = 0
  _update_state!(state0, data, prior, tuners0, rep_aux=rep_aux, fix=fix,
                 flags=flags)

  # State: beta = 1
  _update_state!(state1, data, prior, tuners1, rep_aux=rep_aux, fix=fix,
                 flags=flags)

  # Update beta.
  update_beta_via_pseudo_prior!(state, state0, state1, data, prior)

  # Update theta_beta
  _update_state!(state, data, prior, tuners, rep_aux=rep_aux, fix=fix,
                 flags=flags)
end
