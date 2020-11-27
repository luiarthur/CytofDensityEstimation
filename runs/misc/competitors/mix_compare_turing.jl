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
true_data_dist = cde.Util.SkewT(2, 1, 10, -10)
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
grid = collect(range(-3, 2.5, length=100))
ypdf = pdf.(true_data_dist, grid)
plot(grid, ypdf, label=nothing);
plot!(size=plotsize); ylabel!("density")
savefig(joinpath(imgdir, "data-density.pdf"))
closeall()

# Setup
sims = dict_list(Dict(:K=>collect(1:5), :skew=>[true, false], :tdist=>[true, false]))
getsavedir(sim) = joinpath(resultsdir, savename(sim))
getsavepath(sim) = joinpath(getsavedir(sim), "results.jls")

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
# - [ ] see how many componenets are needed for best fit for each model
# - [ ] see which model is best overall

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
function plot_dic(sims)
  unique_K = unique(getindex.(sims, :K))
  plot(size=plotsize)

  for tdist in [false, true]
    for skew in [false, true]
      sim_subset = filter(s -> s[:tdist]==tdist && s[:skew]==skew, sims)
      dicpaths = [joinpath(getsavedir(s), "img/dic.txt") for s in sim_subset]
      dics = parse.(Float64, [open(f->read(f, String), dicpath)
                              for dicpath in dicpaths])
      plot!(unique_K, dics, label="tdist: $(tdist), skew: $(skew)", lw=3)
    end
  end
  ylabel!("DIC")
  xlabel!("K")
  savefig(joinpath(imgdir, "dic.pdf"))
  closeall()
end
plot_dic(sims)


# Send all results to aws.
cde.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
