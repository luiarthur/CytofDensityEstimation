function litesample(m, spl, N; discard_initial=0, thinning=1, buffersize=50, exclude=[])
  # Run MCMC for `discard_initial` iterations.
  chain = sample(m, spl, 1, save_state=true, discard_initial=discard_initial);

  # Get parameter names.
  all_params = chain.name_map[:parameters]
  excluded_params = collect(Iterators.flatten(namesingroup(chain, p) for p in exclude))
  tracked_params = [filter(x -> !(x in excluded_params), all_params);
                    chain.name_map[:internals]]

  # Vector to store chains.
  cs = []

  # Sample in chunks. Every `buffersize` samples, discard the excluded parameters.
  remainder = mod(N, buffersize)
  num_chunks = let
    nc = div(N, buffersize)
    remainder == 0 ? nc : nc + 1
  end
  for i in ProgressBar(1:num_chunks)
    bs = (i == num_chunks && remainder > 0) ? remainder : buffersize
    chain = resume(chain, bs, save_state=true, thinning=thinning, progress=false)

    # Append current small chain (with only desired parameters) to `cs`.
    append!(cs, [chain[tracked_params]])
  end

  # Return the concatenated chain (with only tracked parameters).
  return cat(cs...)
end
