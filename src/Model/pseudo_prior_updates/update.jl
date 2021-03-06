include("update_beta.jl")


function assumefields!(to::T, from::T) where T
  foreach(fn -> setfield!(to, fn, getfield(from, fn)), fieldnames(T))
end


function update_theta!(state::State, data::Data, prior::Prior,
                       tuners::Tuners; rep_aux::Integer=1,
                       fix::Vector{Symbol}=Symbol[],
                       flags::Vector{Flag}=Flag[])
  update_state!(state, data, prior, tuners,
                fix=[fix; [:p, :beta]], flags=flags)
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
  # Update (β, θᵦ). Though it technically updates β only.
  update_beta_via_pseudo_prior!(state, state0, state1, data, prior)

  # Update θ₀ and θ₁.
  update_theta!(state0, data, prior, tuners0, rep_aux=rep_aux, fix=fix,
                flags=flags)
  update_theta!(state1, data, prior, tuners1, rep_aux=rep_aux, fix=fix,
                flags=flags)

  # Save θᵦ.
  if state.beta
    assumefields!(state, deepcopy(state1))
  else
    assumefields!(state, deepcopy(state0))
  end
end
