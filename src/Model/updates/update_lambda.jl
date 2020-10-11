function update_lambda!(state::State, data::Data, prior::Prior, tuners::Tuners)
  update_lambdaC!(state, data, prior)
  update_lambdaT!(state, data, prior)
end


function update_lambdaC!(state::State, data::Data, prior::Prior)
end


function update_lambdaT!(state::State, data::Data, prior::Prior)
end
