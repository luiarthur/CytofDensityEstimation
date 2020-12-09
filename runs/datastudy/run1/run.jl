println("Compiling on main node ..."); flush(stdout)
include(joinpath(@__DIR__, "imports.jl"))  # For precompile.
println("Done compiling on main node."); flush(stdout)
markers = [:CD3z, :EOMES, :Perforin, :Granzyme_A, :Siglec7, :LAG3, :CD56, :CD57]
# markers = [:CD3z, :EOMES, :Perforin, :Granzyme_A, :Siglec7]
# markers = [:LAG3, :CD56, :CD57]
postprocessing = true

# Read command line args.
if length(ARGS) > 1 || postprocessing
  if postprocessing
    scratchdir = ENV["SCRATCH_DIR"]
    resultsdir = "$(scratchdir)/cde/datastudy/run1"
    awsbucket = "s3://cytof-density-estimation/datastudy/run1"
  else
    resultsdir = ARGS[1]
    awsbucket = ARGS[2]
    println("ARGS: ", ARGS); flush(stdout)
  end
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

using Distributed
println("Load libraries on workers ..."); flush(stdout)
if !istest
  rmprocs(workers())
  addprocs(length(markers)*2)
else
  if nworkers() != 2
    rmprocs(workers())
    addprocs(2)
  end
end
@everywhere include(joinpath(@__DIR__, "imports.jl"))
println("Finished loading libraries on workers."); flush(stdout)


# Read data.
donor = 1
path_to_data = "../../../data/TGFBR2/cytof-data/donor$(donor).csv"
subsample_size = 20000
data = CDE.Util.subsample(DataFrame(CSV.File(path_to_data)), subsample_size,
                          seed=0)

# Helpers.
function make_bucket(marker, beta)
  if awsbucket == nothing
    return awsbucket
  else
    return "$(awsbucket)/donor$(donor)/$(marker)/m$(beta)"
  end
end
make_resdir(marker, beta) = "$(resultsdir)/donor$(donor)/$(marker)/m$(beta)"
make_imgdir(marker, beta) = "$(make_resdir(marker, beta))/img"

# Make configs.
configs = [[let
  yC, yT = CDE.Util.partition(data, marker, log_response=true)
  (yC=yC, yT=yT, K=6, nsamps=nsamps, nburn=nburn, marker=marker,
   awsbucket=make_bucket(marker, beta), p=0.5, beta=beta, thin=1,
   imgdir=make_imgdir(marker, beta),
   resultsdir=make_resdir(marker, beta))
end for beta in 0:1] for marker in markers]

# Parallel run.
if !postprocessing
  println("Starting runs ..."); flush(stdout)
  res = pmap(run, istest ? configs[1] : flatten(configs), on_error=identity)

  println("Status of runs:"); flush(stdout)
  foreach(z -> println(z[2], " => ", z[1]),
          zip(res, getfield.(istest ? configs[1] : flatten(configs), :resultsdir)))
end

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
  postprocess(out0[:chain], out1[:chain], out0[:data], imgdir, bucket,
              density_legend_pos=:topleft, bw_postpred=.3,
              binsC=:auto, binsT=:auto, p=c[1][:p])
end, istest ? [configs[1]] : configs, on_error=identity)

println("Status of BF computation:"); flush(stdout)
foreach(z -> println(z[2], " => ", z[1]),
        zip(res, [c[1][:marker] for c in (istest ? [configs[1]] : configs)]))

println("DONE WITH ALL!"); flush(stdout)

# TMP: Histogram of interesting mu's 
# for marker in (:CD56, :Granzyme_A)
#   ps = []
#   xmin = Inf; xmax = -Inf
#   for beta in (0, 1)
#     resd = make_resdir(marker, beta)
#     imd = make_imgdir(marker, beta)
#     buck = make_resdir(marker, beta)
#     out = BSON.load("$(resd)/results.bson")
#     mu = hcat(getindex.(out[:chain][1], :mu)...)
# 
#     rows = Bool(beta) ? (4,5,6) : (4,5,6)
#     p = plot()
#     for k in rows
#       histogram!(mu[k, :], normalize=true, la=0, palette=:tab10,
#                  label="β=$beta, k=$k", legend=:top)
#       xmin = min(xmin, minimum(mu[k, :]))
#       xmax = max(xmax, maximum(mu[k, :]))
#     end
#     append!(ps, [p])
#   end
#   plot(ps[1], ps[2], size=(400, 300), layout=(2, 1))
#   xlims!(xmin, xmax)
#   savefig(joinpath(imd, "../../img/mu-density.pdf"))
#   closeall()
# end

# Density with only comp4,5 in M0 and comp4,5,6 in M1.
# for marker in [:CD56]
#   ps = []
#   xmin = 2.5; xmax = 8
#   ygrid = range(xmin, xmax, length=100)
#   for beta in (0, 1)
#     resd = make_resdir(marker, beta)
#     imd = make_imgdir(marker, beta)
#     buck = make_resdir(marker, beta)
#     out = BSON.load("$(resd)/results.bson")
#     mu = getindex.(out[:chain][1], :mu)
#     sigma = getindex.(out[:chain][1], :sigma)
#     phi = getindex.(out[:chain][1], :phi)
#     nu = getindex.(out[:chain][1], :nu)
# 
#     rows = Bool(beta) ? (4,5,6) : (4,5)
#     p = plot()
#     for k in rows
#       dens = [pdf.(CDE.Util.SkewT.(mu[b][k], sigma[b][k], nu[b][k], phi[b][k]), ygrid)
#               for b in 1:length(mu)]
#       mean_dens = mean(dens)
#       lower_dens = CDE.MCMC.quantiles(hcat(dens...), .025, dims=2)
#       upper_dens = CDE.MCMC.quantiles(hcat(dens...), .975, dims=2)
#       plot!(ygrid, mean_dens, label=nothing, legend=:topleft, color=:black)
#       plot!(ygrid, lower_dens, fillrange=upper_dens, alpha=.6, label="k=$k, β=$beta")
#     end
#     append!(ps, [p])
#   end
#   plot(ps[1], ps[2], size=(400, 300), layout=(2, 1))
#   xlims!(xmin, xmax)
#   savefig(joinpath(imd, "../../img/post-density-short.pdf"))
#   closeall()
# end
