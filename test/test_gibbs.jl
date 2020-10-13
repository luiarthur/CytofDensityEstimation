mutable struct MyState
  w
  x
  y
  z
end

@testset "Gibbs" begin
  init = MyState(0.0, 0.0, 0.0, 0.0)

  function callback_fn(state, iter, pbar)
    out = nothing
    iter % 20 == 0 && begin
      ll = round(normlogpdf(state.x), digits=3)
      set_description(pbar, "loglike: $(ll)")
      out = Dict(:ll => ll)
    end
    return out
  end

  function update!(state)
    state.w += 1
    state.x = randn()
  end

  chain, laststate, summary_stats = CDE.MCMC.gibbs(init, update!, thin=10,
                                                   monitors=[[:x, :y], [:z]],
                                                   callback_fn=callback_fn,
                                                   nsamps=[500, 10], nburn=500)
  @test length(chain) == 2
  @test length(chain[1]) == 500
  @test length(chain[2]) == 10
  @test length(summary_stats) == 275
  println("Chain start: $(chain[1][1])")
  println("Chain end: $(chain[1][end])")
  println("Last state: $(laststate)")
  println("Head summary stats: $(summary_stats[1])")
  println("Last summary stats: $(summary_stats[end])")
end
