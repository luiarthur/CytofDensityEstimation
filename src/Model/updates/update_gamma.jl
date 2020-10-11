function update_gamma!(state::State, data::Data, prior::Prior, tuners::Tuners)
  update_gammaC!(state, data, prior)
  update_gammaT!(state, data, prior)
end


function update_gammaC!(state::State, data::Data, prior::Prior)
  a, b = params(prior.gamma)
  anew = a + data.ZC + data.ZT * (1 - state.beta)
  bnew = b + data.NC - data.ZC + (data.NT - data.ZT) * (1 - state.beta)
  state.gammaC = rand(Beta(anew, bnew))
end


function update_gammaT!(state::State, data::Data, prior::Prior)
  a, b = params(prior.gamma)
  anew = a + data.ZT * state.beta
  bnew = b + (data.NT - data.ZT) * state.beta
  state.gammaT = rand(Beta(anew, bnew))
end
