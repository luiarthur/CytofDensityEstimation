include("update_p.jl")
include("update_beta.jl")
include("update_gamma.jl")
include("update_eta.jl")
include("update_lambda.jl")
include("update_mu.jl")
include("update_nu.jl")
include("update_omega.jl")
include("update_psi.jl")
include("update_v.jl")
include("update_zeta.jl")

function update_state!(state::State, data::Data, prior::Prior, tuners::Tuners;
                       fix::Vector{Symbol}=Symbol[],
                       flags::Vector{Symbol}=Symbol[])
  # Returns true if a parameter is specified as fixed (to not update).
  isfixed(sym::Symbol) = sym in fix

  # NOTE: Update parameters if they are not held fixed, as specified in `fix`.
  isfixed(:p) || update_p!(state, data, prior)
  isfixed(:beta) || update_beta!(state, data, prior, flags)

  # NOTE: `lambda` must be updated immediately after updating beta since lambda
  # is marginalized over in the update of beta. Marginalizing over lambda makes
  # updates for beta slower, but the mixing will be faster.
  isfixed(:lambda) || update_lambda!(state, data, prior)  

  isfixed(:gamma) || update_gamma!(state, data, prior)
  isfixed(:eta) || update_eta!(state, data, prior)
  isfixed(:mu) || update_mu!(state, data, prior)
  isfixed(:nu) || update_nu!(state, data, prior, tuners)
  isfixed(:omega) || update_omega!(state, data, prior)
  isfixed(:psi) || update_psi!(state, data, prior)
  isfixed(:v) || update_v!(state, data, prior)
  isfixed(:zeta) || update_zeta!(state, data, prior)
end
