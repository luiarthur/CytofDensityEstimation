println("Compile libraries on main processor..."); flush(stdout)
include("imports.jl")
include("scenarios.jl")
println("Finished loading libraries."); flush(stdout)

flatten = Iterators.flatten

if length(ARGS) > 1
  #=
  SCRATCH_DIR = ENV["SCRATCH_DIR"]
  SIMNAME = "sim7"
  RESULTS_DIR = "$(SCRATCH_DIR)/cde/simstudy/$(SIMNAME)/results"
  AWS_BUCKET = "s3://cytof-density-estimation/simstudy/$(SIMNAME)"
  resultsdir = RESULTS_DIR
  awsbucket = AWS_BUCKET
  =#
  resultsdir = ARGS[1]
  awsbucket = ARGS[2]
  println("ARGS: ", ARGS)
  istest = false
  Ni = 10000
  nburn = 6000
  nsamps = 5000
  thin=1
  Ks = [2,4,6]
  p=0.5
else
  resultsdir = "results/test/"
  awsbucket = nothing
  istest = true
  Ni = 5000
  nburn = 1000
  nsamps = 200
  thin = 1
  Ks = [6]
  p = 0.5
end
flush(stdout)

using Distributed
println("Load libraries on workers ..."); flush(stdout)
if !istest
  rmprocs(workers())
  addprocs(12)
  @everywhere include("imports.jl")
else
  if nworkers() != 2
    rmprocs(workers())
    addprocs(2)
  end
  @everywhere include("imports.jl")
end
println("Finished loading libraries on workers."); flush(stdout)

# Helpers.
simname(K, snum, beta) = "K$(K)/scenario$(snum)/m$(beta)"
function make_bucket(K, snum, beta, bucket)
  return (bucket == nothing) ? bucket : "$(bucket)/$(simname(K, snum, beta))"
end
make_resdir(K, snum, beta) = "$(resultsdir)/$(simname(K, snum, beta))"
function make_imgdir(K, snum, beta)
  resdir = make_resdir(K, snum, beta)
  imgdir = "$(resdir)/img"
  mkpath(imgdir)
  return imgdir
end

configs = [[let
  # NOTE
  simdata = scenarios(snum, seed=1, Ni=Ni)
  bucket = make_bucket(K, snum, beta, awsbucket)
  resdir = make_resdir(K, snum, beta)
  imgdir = make_imgdir(K, snum, beta)
  (awsbucket=bucket, simdata=simdata, resultsdir=resdir, imgdir=imgdir,
   snum=snum, beta=beta, K=K, nsamps=nsamps, thin=thin, nburn=nburn, p=p)
end for beta in 0:1] for K in Ks for snum in 1:4]


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
        zip(res, [c[1][:snum] for c in (istest ? [configs[1]] : configs)]))

println("DONE WITH ALL!"); flush(stdout)
