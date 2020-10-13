#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#
# TODO: ApproximateTwoSampleKSTest(a, b)

@testset "Test fit" begin
  simdata = CDE.Model.generate_samples(; NC=1000, NT=999, gammaC=.3, gammaT=.3,
                                       etaC=[.5, .5, 0], etaT=[.5, .4, .1],
                                       loc=[-1., 1, 2], scale=[.7, 1.3, 1],
                                       df=[15, 30, 10], skew=[-2, -5, 0])
end
