include("mixst_model.jl")
using DrWatson
using Serialization
using ProgressBars

plotsize = (400, 400)
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)

scratchdir = ENV["SCRATCH_DIR"]
resultsdir = joinpath(scratchdir, "cde", "misc", "competitors")
awsbucket = "s3://cytof-density-estimation/misc/competitors"

## Generate data.
Random.seed!(0);
true_data_dist = cde.Util.SkewT(2, 1, 7, -10)
y = rand(true_data_dist, 1000);

# Plot data.
imgdir = joinpath(resultsdir, "img") 
mkpath(imgdir)

# Data Histogram.
histogram(y, bins=100, label=nothing, la=0);
plot!(size=plotsize); ylabel!("density")
savefig(joinpath(imgdir, "data-hist.pdf"))
closeall()

# True data density.
grid = collect(range(-4.5, 2.5, length=100))
ypdf = pdf.(true_data_dist, grid)
plot(grid, ypdf, label=nothing);
plot!(size=plotsize); ylabel!("density")
savefig(joinpath(imgdir, "data-density.pdf"))
closeall()

# Setup
sims = dict_list(Dict(:K=>collect(1:5), :skew=>[true, false], :tdist=>[true, false]))
getsavedir(sim) = joinpath(resultsdir, savename(sim))
getsavepath(sim) = joinpath(getsavedir(sim), "results.jls")

Random.seed!(1);
for sim in sims
  savedir = getsavedir(sim)
  mkpath(savedir)
  savepath = getsavepath(sim)

  K = sim[:K]
  skew = sim[:skew]
  tdist = sim[:tdist]

  v, zeta = make_aux(y)
  m = MixST(y, K, v, zeta)
  spl = make_sampler(y, K, v, zeta, skew=skew, tdist=tdist)
  burn, nsamps = 5000, 3000
  chain = sample(m, spl, burn + nsamps, save_state=true)[(burn+1):end];

  serialize(savepath, chain)

  # NOTE: Load via
  # deserialize(getsavepath(sim))
end

# Send results to aws.
cde.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)

# Postprocess
# TODO:
# - [X] plot data (observed and truth).
# - [X] plot psterior estimate and point-wise ci.
# - [ ] get posterior distributions and trace of each parameter.
# - [X] compute dic for all models.
# - [X] see how many componenets are needed for best fit for each model
# - [X] see which model is best overall

function postprocess(sim)
  savepath = getsavepath(sim)
  savedir = getsavedir(sim)
  chain = deserialize(savepath);
  plot_posterior(chain, savedir, y, grid, true_data_dist)
end

for sim in ProgressBar(sims)  # TODO: remove index
  postprocess(sim)
end

# Plot DIC for all models.
function plot_metrics(sims, colors=palette(:tab10),
                      marker=[:rect, :circle, :utriangle, :pentagon])
  unique_K = unique(getindex.(sims, :K))

  for metric in ["dic", "mean_deviance"]
    plot(size=plotsize)
    plotidx = 0
    for tdist in [false, true]
      for skew in [false, true]
        plotidx +=1
        sim_subset = filter(s -> s[:tdist]==tdist && s[:skew]==skew, sims)

        metric_paths = [joinpath(getsavedir(s), "img/$(metric).txt")
                        for s in sim_subset]
        metrics = parse.(Float64, [open(f->read(f, String), metric_path)
                                   for metric_path in metric_paths])
        plot!(unique_K, metrics, label="tdist: $(tdist), skew: $(skew)",
              marker=marker[plotidx], color=:grey, ms=6)
      end
    end
    ylabel!(replace(metric, "_" => " "))
    xlabel!("K")
    savefig(joinpath(imgdir, "$(metric).pdf"))
    closeall()
  end
end
plot_metrics(sims)


# Send all results to aws.
cde.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
