function default_callback_fn(state::StateAM, data::Data, prior::PriorAM,
                             iter::Int, pbar::MCMC.ProgressBars.ProgressBar)
  ll = round(loglike(state, data, prior.p), digits=3)
  MCMC.ProgressBars.set_postfix(pbar, loglike=ll)
  return Dict(:loglike => ll)
end


function amfit(init::StateAM, data::Data, prior::PriorAM;
               tuner::Union{Nothing, MCMC.MvTuner}=nothing,
               nsamps::Vector{Int}=[1000], nburn::Int=1000, thin::Int=1,
               seed=nothing, verbose::Int=1, temper::Real=1)
  seed == nothing || Random.seed!(seed)
  tuner == nothing && (tuner = MCMC.MvTuner(6 * prior.K))

  # Print settings for sanity check.
  if verbose > 0
    println("seed: ", seed)
  end

  function update!(state::StateAM)
    v = tovec(state)
    function logprob(v::Vector{Float64})
      s = deepcopy(state)
      fromvec!(s, v)
      ll = loglike(s, data, prior.p) / temper
      lp = logprior(s, prior)
      lj = logabsjacobian(v, prior)
      return ll + lp + lj
    end
    newv = MCMC.metropolisAdaptive(v, logprob, tuner)
    newv != v && fromvec!(state, newv)
  end

  function _callback_fn(state::StateAM, iter::Int,
                        pbar::MCMC.ProgressBars.ProgressBar)
    return default_callback_fn(state, data, prior, iter, pbar)
  end

  chain, laststate, summarystats = MCMC.gibbs(
      init, update!, nsamps=nsamps, nburn=nburn, thin=thin,
      monitors=[[:gammaC, :gammaT, :etaC, :etaT, :mu, :sigma, :nu, :phi]],
      callback_fn=_callback_fn)

  return chain, laststate, summarystats, tuner
end
