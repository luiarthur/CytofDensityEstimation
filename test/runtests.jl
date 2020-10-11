#= Load these if running these tests in terminal
import Pkg; Pkg.activate("../")  # CytofDensityEstimation
include("runtests.jl")
=#
println("Testing ...")

using Test
using CytofDensityEstimation
using StatsFuns
using Statistics

include("test_gibbs.jl")
