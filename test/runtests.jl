#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#
println("Testing ...")

using Test
using CytofDensityEstimation
using Distributions
using StatsFuns
using Statistics
using ProgressBars
using StatsPlots

# using StatsPlots
# NOTE: Might need to set
# ENV["GRDIR"] = ""
# ENV["GKSwstype"] = "nul"

const CDE = CytofDensityEstimation

include("test_skewt.jl")
include("test_gibbs.jl")
include("test_updates.jl")
include("test_fit.jl")
include("test_fit_via_pseudo_prior.jl")
include("test_generate_sample.jl")
include("test_mcmc.jl")
