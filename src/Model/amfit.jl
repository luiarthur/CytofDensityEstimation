function default_callback_fn(state::StateAM, data::Data, prior::PriorAM,
                             iter::Int, pbar::MCMC.ProgressBars.ProgressBar)
  ll = round(loglike(state, data, prior.beta), digits=3)
  MCMC.ProgressBars.set_postfix(pbar, loglike=ll)
  return Dict(:loglike => ll)
end


function amfit(init::StateAM, data::Data, prior::PriorAM;
               tuner::Union{Nothing, MCMC.MvTuner}=nothing,
               nsamps::Vector{Int}=[1000], nburn::Int=1000, thin::Int=1,
               seed=nothing, verbose::Int=1, temper::Real=1, propcov=nothing,
               nc::Int=0, nt::Int=0, openblas_num_threads=1)

  LinearAlgebra.BLAS.set_num_threads(openblas_num_threads)
  seed == nothing || Random.seed!(seed)
  tuner == nothing && (tuner = MCMC.MvTuner(tovec(init)))

  # Print settings for sanity check.
  if verbose > 0
    println("seed: ", seed)
    println("temper: ", temper)
    println("openblas_num_threads: ", openblas_num_threads)
  end

  function update!(state::StateAM)
    _data = subsample_data(data, nc, nt)
    v = tovec(state)
    function logprob(v::Vector{Float64})
      s = deepcopy(state)
      fromvec!(s, v)
      ll = loglike(s, _data, prior.beta) * (data.NC + data.NT) / (_data.NC + _data.NT) / temper
      lp = logprior(s, prior)
      lj = logabsjacobian(v, prior)
      return ll + lp + lj
    end
    if propcov == nothing
      newv = MCMC.metropolisAdaptive(v, logprob, tuner)
    else
      newv = MCMC.metropolis(v, logprob, propcov)
    end
    newv != v && fromvec!(state, newv)
  end

  function _callback_fn(state::StateAM, iter::Int,
                        pbar::MCMC.ProgressBars.ProgressBar)
    # return default_callback_fn(state, data, prior, iter, pbar)
  end

  chain, laststate, summarystats = MCMC.gibbs(
      init, update!, nsamps=nsamps, nburn=nburn, thin=thin,
      monitors=[[:gammaC, :gammaT, :etaC, :etaT, :mu, :sigma, :nu, :phi]],
      callback_fn=_callback_fn)

  return chain, laststate, summarystats, tuner
end
