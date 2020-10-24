# TODO: Check.

include("update_beta.jl")

function update_theta!(state::State, data::Data, prior::Prior,
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
  # Update β
  update_beta_via_pseudo_prior!(state, state0, state1, data, prior,
                                tuners, tuners0, tuners1, rep_aux=rep_aux,
                                fix=fix, flags=flags)

  # FIXME: `|| 0.01 > rand()` is a hack. Any fix?
  # Update θᵦ. And occasionally update θ_{1-beta} anyway.
  if state.beta || 0.01 > rand()
    update_theta!(state1, data, prior, tuners1, rep_aux=rep_aux, fix=fix,
                  flags=flags)
    assumefields!(state, deepcopy(state1))
  end
  if !state.beta || 0.01 > rand()
    update_theta!(state0, data, prior, tuners0, rep_aux=rep_aux, fix=fix,
                  flags=flags)
    assumefields!(state, deepcopy(state0))
  end
end
