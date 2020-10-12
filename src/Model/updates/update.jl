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

function update_state!(state::State, data::Data, prior::Prior, tuners::Tuners,
                       flags::Vector{Symbol}=Symbol[])
  update_p!(state, data, prior)
  update_beta!(state, data, prior, flags)
  # NOTE: `lambda` must be updated immediately after updating beta since lambda
  # is marginalized over in the update of beta. Marginalizing over lambda makes
  # updates for beta slower, but the mixing will be faster.
  update_lambda!(state, data, prior)  
  update_gamma!(state, data, prior)
  update_eta!(state, data, prior)
  update_mu!(state, data, prior)
  update_nu!(state, data, prior, tuners)
  update_omega!(state, data, prior)
  update_psi!(state, data, prior)
  update_v!(state, data, prior)
  update_zeta!(state, data, prior)
end
