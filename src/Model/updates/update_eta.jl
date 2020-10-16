function update_eta!(state::State, data::Data, prior::Prior)
  update_etaC!(state, data, prior)
  update_etaT!(state, data, prior)
end


function update_etaC!(state::State, data::Data, prior::Prior)
  anew = prior.eta.alpha .+ 0

  for n in eachindex(state.lambdaC)
    anew[state.lambdaC[n]] += 1
  end

  if !state.beta
    for n in eachindex(state.lambdaT)
      anew[state.lambdaT[n]] += 1
    end
  end

  state.etaC = rand(Dirichlet(anew))
end


function update_etaT!(state::State, data::Data, prior::Prior)
  anew = prior.eta.alpha .+ 0

  if state.beta
    for n in eachindex(state.lambdaT)
      anew[state.lambdaT[n]] += 1
    end
  end

  state.etaT = rand(Dirichlet(anew))
end
