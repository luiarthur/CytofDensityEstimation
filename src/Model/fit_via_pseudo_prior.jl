# TODO: Check. Document.
"""
x: the running means
y: the new values
"""
function update_running_mean!(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                              iter::Integer, nburn::Real)
  iter == nburn && (x .= 0)
  if iter <= nburn
    x .= x * (iter - 1) / iter + y / iter
  else
    update_running_mean!(x, y, iter - nburn, Inf)
  end
end


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
- `flags::Vector{Flag}`: Available options include `UpdateBetaWithSkewT()`
   and `UpdateLambdaWithSkewT()`.
   Defaults to `default_flags()`.
- `verbose::Int`: How much info to print. Defaults to 1.
- `rep_aux::Int`: How many times to update `zeta, v, lambda` each iteration.
   Defaul=1.
- `callback_fn`: A function with signature `(state::State, data::Data,
   prior::Prior, iter::Int, pbar::ProgressBar)` that returns a dicionary of
   summary statistics. Enables one to add metrics to progress bar.
"""
function fit_via_pseudo_prior(init::State, data::Data, prior::Prior,
                              tuners::Tuners;
                              nsamps::Vector{Int}=[1000], nburn::Int=1000,
                              thin::Int=1, warmup::Integer=10,
                              monitors::Vector{Vector{Symbol}}=default_monitors(),
                              callback_fn::Function=default_callback_fn,
                              fix::Vector{Symbol}=Symbol[],
                              rep_aux::Integer=1, seed=nothing,
                              flags::Vector{Flag}=default_flags(),
                              verbose::Int=1)
  seed == nothing || Random.seed!(seed)

  fix = unique(fix)
  flags = unique(flags)

  println("seed: ", seed)
  println("flags: ", flags)
  println("fix: ", fix)
  println("monitors: ", monitors)

  isfixed(sym::Symbol) = sym in fix

  state0 = deepcopy(init); state0.beta = false
  state1 = deepcopy(init); state1.beta = true
  tuners0 = deepcopy(tuners)
  tuners1 = deepcopy(tuners)

  # Warmup phase for state0, state1.
  for _ in MCMC.ProgressBars.ProgressBar(1:warmup)
    update_theta!(state1, data, prior, tuners1, rep_aux=rep_aux, fix=fix,
                  flags=flags)
    update_theta!(state0, data, prior, tuners0, rep_aux=rep_aux, fix=fix,
                  flags=flags)
    flush(stdout)
  end

  # Define update function.
  function update!(state)
    isfixed(:p) || update_p!(state, data, prior)
    curr_p = state.p
    update_state_via_pseudo_prior!(state, state0, state1, data, prior, tuners,
                                   tuners0, tuners1, rep_aux=rep_aux, fix=fix,
                                   flags=flags)
    state.p = curr_p
  end

  # Initialize beta_hat tracker.
  beta_hat = [0.0]

  # Define callback.
  function _callback_fn(state::State, iter::Int,
                        pbar::MCMC.ProgressBars.ProgressBar)
    update_running_mean!(beta_hat, [state.beta], iter, nburn)
    cb_out = callback_fn(state, data, prior, iter, pbar)
    pbar.postfix = (beta_hat=round(beta_hat[1], digits=3), pbar.postfix...)
    return cb_out
  end

  # Run Gibbs sampler.
  chain, laststate, summarystats = MCMC.gibbs(
      init, update!, nsamps=nsamps, nburn=nburn, thin=thin, monitors=monitors,
      callback_fn=_callback_fn)

  return chain, laststate, summarystats
end


ppfit = fit_via_pseudo_prior
