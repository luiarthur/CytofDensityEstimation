include(joinpath(@__DIR__, "imports.jl"))  # For precompile.
markers = [:CD3z, :EOMES, :Perforin, :Granzyme_A, :Siglec7]

using Distributed
rmprocs(workers())
addprocs(length(markers))
@everywhere include(joinpath(@__DIR__, "imports.jl"))

# Read command line args.
if length(ARGS) > 1
  resultsdir = ARGS[1]
  awsbucket = ARGS[2]
  println("ARGS: ", ARGS); flush(stdout)
  nsamps = 10000
  nburn = 2000
else
  resultsdir = "results/test/"
  awsbucket = nothing
  nsamps = 10
  nburn = 20
end

# Read data.
donor = 1
path_to_data = "../../../data/TGFBR2/cytof-data/donor$(donor).csv"
subsample_size = 20000
data = CDE.Util.subsample(DataFrame(CSV.File(path_to_data)), subsample_size,
                          seed=0)
configs = [let
             _awsbucket = (awsbucket == nothing) ? awsbucket : "$(awsbucket)/donor$(donor)/$(marker)"
             yC, yT = CDE.Util.partition(data, marker)
             (yC=yC, yT=yT, K=5, nsamps=nsamps, nburn=nburn, marker=marker,
              awsbucket=_awsbucket,
              resultsdir=joinpath(resultsdir, "donor$(donor)/$(marker)"))
           end for marker in markers]

# Parallel run.
println("Starting runs ...")
res = pmap(run, configs, on_error=identity)

println("Status of runs:")
foreach(z -> println(z[2], " => ", z[1]),
        zip(res, getfield.(configs, :resultsdir)))

println("DONE WITH ALL!"); flush(stdout)
