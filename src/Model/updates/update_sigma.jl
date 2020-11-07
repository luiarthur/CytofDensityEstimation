function update_sigma!(state::State)
  state.sigma .= Util.scalefromaltskewt.(sqrt.(state.omega), state.psi)
end
