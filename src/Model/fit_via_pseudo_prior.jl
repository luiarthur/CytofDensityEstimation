# TODO: Check. Document.

"""
- `init::State`: Initial state.
- `data::Data`: Data.
- `prior::Prior`: Priors.
- `tuners::Tuners`: Tuners.
- `nsamps::Vector{Int}`: Number of posterior samples for each monitor. Needs to
   be in descending order. Defaults to `[1000]`.
- `nburn::Int`: Number of burn-in iterations. Defaults to `1000`.
- `thin::Int`: Thinning factor for first monitor. Defaults to `1`.
- `monitors::Vector{Vector{Symbol}}`: Parameters to keep samples of. Defaults
   to `default_monitors()`.
- `fix::Vector{Symbol}`: Parameters to hold constant. Defaults to `Symbol[]`.
- `flags::Vector{Flag}`: Available options include `:update_beta_with_skewt`
   and `:update_lambda_with_skewt`.
   Defaults to `default_flags()`.
- `verbose::Int`: How much info to print. Defaults to 1.
- `callback_fn`: A function with signature `(state::State, data::Data,
   prior::Prior, iter::Int, pbar::ProgressBar)` that returns a dicionary of
   summary statistics. Enables one to add metrics to progress bar.
"""
function fit_via_pseudo_prior(init::State, data::Data, prior::Prior,
                              tuners::Tuners, p::Real;
                              nsamps::Vector{Int}=[1000], nburn::Int=1000,
                              thin::Int=1,
                              monitors::Vector{Vector{Symbol}}=default_monitors(),
                              callback_fn::Function=default_callback_fn,
                              fix::Vector{Symbol}=Symbol[], rep0::Integer=1,
                              rep1::Integer=1, rep_aux::Integer=1,
                              seed=nothing,
                              flags::Vector{Flag}=default_flags(),
                              verbose::Int=1)

  @assert 0 <= p <= 1
  seed == nothing || Random.seed!(seed)

  println("seed: ", seed)
  println("flags: ", flags)
  println("rep0: ", rep0)
  println("rep1: ", rep1)
  println("fix: ", fix)
  println("monitors: ", monitors)

  isfixed(sym::Symbol) = sym in fix

  init.p = p
  state0 = deepcopy(init); state0.beta = false
  state1 = deepcopy(init); state1.beta = true
  tuners0 = deepcopy(tuners)
  tuners1 = deepcopy(tuners)

  function update!(state)
    update_state_via_pseudo_prior!(state, state0, state1, 
                                   data, prior, tuners0, tuners1,
                                   rep0=rep0, rep1=rep1, rep_aux=rep_aux,
                                   fix=fix, flags=flags)
  end

  function _callback_fn(state::State, iter::Int,
                        pbar::MCMC.ProgressBars.ProgressBar)
    return callback_fn(state, data, prior, iter, pbar)
  end

  chain, laststate, summarystats = MCMC.gibbs(
      init, update!, nsamps=nsamps, nburn=nburn, thin=thin, monitors=monitors,
      callback_fn=_callback_fn)

  return chain, laststate, summarystats
end


ppfit = fit_via_pseudo_prior
