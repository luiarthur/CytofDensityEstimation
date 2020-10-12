mutable struct MyState
  x
end

@testset "Gibbs" begin
  init = MyState(0.0)
  loglike = []
  function update!(state, i)
    state.x = randn()
    append!(loglike, normlogpdf(state.x))
    sleep(.001)
  end
  out, laststate = CDE.MCMC.gibbs(init, update!, thins=[10], monitors=[[:x]],
                                  nmcmc=500, nburn=500)
  println(laststate)
  # println(out[1])
end