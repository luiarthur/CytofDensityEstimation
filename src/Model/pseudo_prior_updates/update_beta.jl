function assumefields!(to::T, from::T) where T
  foreach(fn -> setfield!(to, fn, getfield(from, fn)), fieldnames(T))
end

# TODO. Check.
function update_beta_via_pseudo_prior!(state::State,
                                       state0::State, state1::State, 
                                       data::Data, prior::Prior,
                                       tuners::Tuners,
                                       tuners0::Tuners, tuners1::Tuners;
                                       rep_aux::Integer=1,
                                       fix::Vector{Symbol}=Symbol[],
                                       flags::Vector{Flag}=Flag[])
  # Create proposed state.
  pstate0 = deepcopy(state0)
  pstate1 = deepcopy(state1)
  if state.beta
    update_theta!(pstate1, data, prior, tuners1, rep_aux=rep_aux, fix=fix,
                  flags=flags)
  else
    update_theta!(pstate0, data, prior, tuners0, rep_aux=rep_aux, fix=fix,
                  flags=flags)
  end


  lp0 = marginal_logprior(pstate0, prior)
  lp1 = marginal_logprior(pstate1, prior)
  ll0 = marginal_loglike(data, Dict(:mu=>pstate0.mu, :omega=>pstate0.omega,
                                    :psi=>pstate0.psi, :nu=>pstate0.nu,
                                    :beta=>pstate0.beta,
                                    :etaC=>pstate0.etaC, :gammaC=>pstate0.gammaC,
                                    :etaT=>pstate0.etaT, :gammaT=>pstate0.gammaT))
  ll1 = marginal_loglike(data, Dict(:mu=>pstate1.mu, :omega=>pstate1.omega,
                                    :psi=>pstate1.psi, :nu=>pstate1.nu,
                                    :beta=>pstate1.beta,
                                    :etaC=>pstate1.etaC, :gammaC=>pstate1.gammaC,
                                    :etaT=>pstate1.etaT, :gammaT=>pstate1.gammaT))

  log_acceptance_ratio_1 = (lp1 + ll1) - (lp0 + ll0) + logit(state.p)
  log_acceptance_ratio = state.beta ? -log_acceptance_ratio_1 : log_acceptance_ratio_1

  if log_acceptance_ratio > log(rand())
    if state.beta
      assumefields!(state, deepcopy(pstate0))
      assumefields!(state0, deepcopy(pstate0))
    else
      assumefields!(state, deepcopy(pstate1))
      assumefields!(state1, deepcopy(pstate1))
    end
  end
end
