include("mixst_model.jl")
using DrWatson
using Serialization
resultsdir = "results"
awsbucket = "s3://cytof-density-estimation/compare-models"

## Generate data.
Random.seed!(0);
y = rand(cde.Util.SkewT(2, 1, 10, -10), 1000);

# Plot data.
# histogram(y, bins=100, label=nothing, la=0)

# Setup
sims = dict_list(Dict(:K=>[5], :skew=>[true, false], :tdist=>[true, false]))
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
  burn, nsamps = 2000, 2000
  chain = sample(m, spl, burn + nsamps, save_state=true)[(burn+1):end];

  serialize(savepath, chain)

  # NOTE: Load via
  # deserialize(getsavepath(sim))
end

# Send results to aws.
cde.Util.s3sync(from=resultsdir, to=awsbucket, tags=`--exclude '*.nfs'`)
