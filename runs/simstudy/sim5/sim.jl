println("Compile libraries on main processor..."); flush(stdout)
include("imports.jl")
include("scenarios.jl")
println("Finished loading libraries."); flush(stdout)

flatten = Iterators.flatten

if length(ARGS) > 1
  resultsdir = ARGS[1]
  awsbucket = ARGS[2]
  println("ARGS: ", ARGS)
  istest = false
  Ni = 1000
  nburn = 3000
  nsamps = 3000
  K = 6
else
  resultsdir = "results/test/"
  awsbucket = nothing
  istest = true
  Ni = 100
  nburn = 300
  nsamps = 300
  K = 2
end
imgdir = joinpath(resultsdir, "img")
mkpath(imgdir)
flush(stdout)

using Distributed
println("Load libraries on workers ..."); flush(stdout)
if !istest
  rmprocs(workers())
  addprocs(8)
  @everywhere include("imports.jl")
end
println("Finished loading libraries on workers."); flush(stdout)

# Helpers.
simname(snum, beta) = "scenario$(snum)/m$(beta)"
function make_bucket(snum, beta, bucket)
  return (bucket == nothing) ? bucket : "$(bucket)/$(simname(snum, beta))"
end
make_resdir(snum, beta) = "$(resultsdir)/$(simname(snum, beta))"
function make_imgdir(snum, beta)
  resdir = make_resdir(snum, beta)
  imdir = "$(resdir)/img"
  mkpath(imdir)
  return imdir
end

configs = [[let
  # NOTE
  _awsbucket = make_bucket(snum, beta, awsbucket)
  simdata = scenarios(snum, seed=1, Ni=Ni)
  resdir = make_resdir(snum, beta)
  _imgdir = make_imgdir(snum, beta)
  (awsbucket=_awsbucket, simdata=simdata, resultsdir=resdir, imgdir=_imgdir,
   snum=snum, beta=beta, K=K, nsamps=nsamps, nburn=nburn, p=0.5)
end for beta in 0:1] for snum in 1:4]


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
  @assert length(unique(getfield.(c, :snum))) == 1
  snum = c[1][:snum]
  out0 = BSON.load("$(resultsdir)/scenario$(snum)/m0/results.bson")
  out1 = BSON.load("$(resultsdir)/scenario$(snum)/m1/results.bson")
  imdir = "$(imgdir)/scenario$(snum)/img"
  bucket = awsbucket == nothing ? awsbucket : "$(awsbucket)/scenario$(snum)"
  postprocess(out0[:chain], out1[:chain], out0[:data], 
              imdir, bucket, simdata=c[1][:simdata],
              density_legend_pos=:topleft)
end, istest ? [configs[1]] : configs, on_error=identity)

println("Status of BF computation:"); flush(stdout)
foreach(z -> println(z[2], " => ", z[1]),
        zip(res, [c[1][:snum] for c in (istest ? [configs[1]] : configs)]))

println("DONE WITH ALL!"); flush(stdout)
