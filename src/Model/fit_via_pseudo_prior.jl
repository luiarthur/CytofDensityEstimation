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
                              tuners::Tuners; p::Real,
                              nsamps::Vector{Int}=[1000], nburn::Int=1000,
                              thin::Int=1,
                              monitors::Vector{Vector{Symbol}}=default_monitors(),
                              callback_fn::Function=default_callback_fn,
                              fix::Vector{Symbol}=Symbol[],
                              rep_aux::Integer=1, seed=nothing,
                              flags::Vector{Flag}=default_flags(),
                              verbose::Int=1)

  @assert 0 <= p <= 1
  seed == nothing || Random.seed!(seed)

  println("seed: ", seed)
  println("flags: ", flags)
  println("fix: ", fix)
  println("monitors: ", monitors)

  isfixed(sym::Symbol) = sym in fix

  init.p = p
  state0 = deepcopy(init); state0.beta = false
  state1 = deepcopy(init); state1.beta = true
  tuners0 = deepcopy(tuners)
  tuners1 = deepcopy(tuners)

  function update!(state)
    update_state_via_pseudo_prior!(state, state0, state1, data, prior,
                                   tuners0, tuners1, rep_aux=rep_aux, fix=fix,
                                   flags=flags)
  end

  beta_hat = [0.0]
  function _callback_fn(state::State, iter::Int,
                        pbar::MCMC.ProgressBars.ProgressBar)
    update_running_mean!(beta_hat, [state.beta], iter, nburn)
    cb_out = callback_fn(state, data, prior, iter, pbar)
    MCMC.ProgressBars.set_postfix(pbar, beta_hat=round(beta_hat[1], digits=3),
                                  loglike=cb_out[:loglike])
    return cb_out
  end

  chain, laststate, summarystats = MCMC.gibbs(
      init, update!, nsamps=nsamps, nburn=nburn, thin=thin, monitors=monitors,
      callback_fn=_callback_fn)

  return chain, laststate, summarystats
end


ppfit = fit_via_pseudo_prior
