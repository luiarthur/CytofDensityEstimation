# TODO:
# - [ ] add documentation.

default_monitors() = [[:p, :beta, :gammaC, :gammaT, :etaC, :etaT, :mu, :nu,
                      :omega, :psi]]

complete_monitors() = [default_monitors()[1],
                      [:lambdaC, :lambdaT, :vC, :vT, :zetaC, :zetaT]]

default_flags() = [:update_beta_with_latent_var_like]

function default_callback_fn(state::State, data::Data, prior::Prior,
                             iter::Int, pbar::MCMC.ProgressBars.ProgressBar)
  ll = loglike_latent_var(state, data)
  MCMC.ProgressBars.set_description(pbar, string(Printf.@sprintf("loglike: %.3e", ll)))
  return Dict(:loglike => ll)
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
- `flags::Vector{Symbol}`: Available options include `:update_beta_with_latent_var_like`.
   Defaults to `default_flags()`.
- `verbose::Int`: How much info to print. Defaults to 1.
- `callback_fn`: A function with signature `(state::State, data::Data,
   prior::Prior, iter::Int, pbar::ProgressBar)` that returns a dicionary of
   summary statistics. Enables one to add metrics to progress bar.
"""
function fit(init::State, data::Data, prior::Prior, tuners::Tuners;
             nsamps::Vector{Int}=[1000], nburn::Int=1000, thin::Int=1,
             monitors::Vector{Vector{Symbol}}=default_monitors(),
             callback_fn::Function=default_callback_fn,
             fix::Vector{Symbol}=Symbol[],
             flags::Vector{Symbol}=default_flags(), verbose::Int=1)
  function update!(state)
    update_state!(state, data, prior, tuners, fix=fix, flags=flags)
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
