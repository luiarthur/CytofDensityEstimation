function update_phi!(state::State)
  state.phi .= Util.skewfromaltskewt.(sqrt.(state.omega), state.psi)
end
