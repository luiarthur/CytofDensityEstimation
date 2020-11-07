include(joinpath(@__DIR__, "imports.jl"))  # For precompile.
markers = [:CD3z, :EOMES, :Perforin, :Granzyme_A, :Siglec7]

using Distributed
rmprocs(workers())
addprocs(length(markers) * 2)
@everywhere include(joinpath(@__DIR__, "imports.jl"))

# Read command line args.
if length(ARGS) > 1
  resultsdir = ARGS[1]
  awsbucket = ARGS[2]
  println("ARGS: ", ARGS); flush(stdout)
  nsamps = 5000
  nburn = 6000
  istest=false
else
  resultsdir = "results/test/"
  awsbucket = nothing
  nsamps = 10
  nburn = 20
  istest=true
end

# Read data.
donor = 1
path_to_data = "../../../data/TGFBR2/cytof-data/donor$(donor).csv"
subsample_size = 20000
data = CDE.Util.subsample(DataFrame(CSV.File(path_to_data)), subsample_size,
                          seed=0)
configs = [[let
  bucket = (awsbucket == nothing) ? awsbucket : "$(awsbucket)/donor$(donor)/$(marker)"
  yC, yT = CDE.Util.partition(data, marker, log_response=true)
  (yC=yC, yT=yT, K=6, nsamps=nsamps, nburn=nburn, marker=marker,
   awsbucket=bucket, p=0.5, beta=beta, thin=1, 
   resultsdir=joinpath(resultsdir, "donor$(donor)/$(marker)"))
end for beta in 0:1] for marker in markers]

# Parallel run.
println("Starting runs ..."); flush(stdout)
res = pmap(run, istest ? configs[1] : flatten(configs), on_error=identity)

println("Status of runs:"); flush(stdout)
foreach(z -> println(z[2], " => ", z[1]),
        zip(res, getfield.(istest ? configs[1] : flatten(configs), :resultsdir)))

# Bayes factor
println("Compute BF:"); flush(stdout)
res = pmap(c -> let
  # NOTE
  @assert length(unique(getfield.(c, :marker))) == 1
  resd0 = c[1][:resultsdir]
  resd1 = c[2][:resultsdir]
  out0 = BSON.load("$(resd0)/results.bson")
  out1 = BSON.load("$(resd1)/results.bson")
  imgdir = "$(c[1][:imgdir])/../../img"; mkpath(imgdir)
  bucket = c[1][:awsbucket] == nothing ? nothing : "$(c[1][:awsbucket])/../img"
  postprocess(out0[:chain], out1[:chain], out0[:data], 
              imgdir, bucket, simdata=c[1][:simdata],
              density_legend_pos=:topleft, bw_postpred=.3, binsC=50, binsT=100,
              p=c[1][:p])
end, istest ? [configs[1]] : configs, on_error=identity)

println("Status of BF computation:"); flush(stdout)
foreach(z -> println(z[2], " => ", z[1]),
        zip(res, [c[1][:marker] for c in (istest ? [configs[1]] : configs)]))

println("DONE WITH ALL!"); flush(stdout)
