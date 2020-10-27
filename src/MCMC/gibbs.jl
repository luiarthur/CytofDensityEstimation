# TODO:
# - Refactor:
#     - `Vector{Symbol}` in many instances should be replaced with the more
#       appropriate `Set{Symbol}`.

monitors_default() = Vector{Vector{Symbol}}([])


"""
Default callback function for `gibbs` does nothing. You can provide your own
callback which has the same signature. One use for this is computing summary
statistics like log-likelihood after each MCMC iteration. If summary statistics
are returned, they will be appended and returned at the end of the MCMC.

For example,
```julia
function callback_fn(state, iter, pbar)
  out = nothing
  if iter % 10 == 0
    ll = round(compute_loglike(state), digits=3)
    set_description(pbar, "loglike: \$(ll)")
    out = Dict(:ll => ll)
  end

  return out
end
```
"""
function default_callback_fn(state::T, iter::Int, pbar::ProgressBar) where T
  return
end


"""
Checks if a field is a subtype.
If a field has the form "a__b", the name
of the field is `a`, and it has a field `b`.
"""
issubtype(x::Symbol)::Bool = occursin("__", String(x))


function deepcopyFields(state::T, fields::Vector{Symbol}) where T
  substate = Dict{Symbol, Any}()

  # Partition fields into subtypes and regular fields
  subtypefields, regfields = partition(issubtype, fields)

  # Get level1 fields
  for field in regfields
    substate[field] = deepcopy(getfield(state, field))
  end

  # Get level2 fields
  for field in subtypefields
    statename, _field = split(String(field), "__")
    _state = getfield(state, Symbol(statename))
    substate[field] = deepcopy(getfield(_state, Symbol(_field)))
  end

  return substate
end


showtime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")


function gibbs(init::T, update!::Function;
               monitors::Vector{Vector{Symbol}}=monitors_default(),
               nsamps::Vector{Int}=[1000], nburn::Int=0,
               thin::Int=1, callback_fn::Function=default_callback_fn,
               verbose::Int=1) where T
  # Assert that the first monitor is the primary one.
  # The first sample should collect the most samples.
  @assert issorted(nsamps, rev=true)

  # Make a copy of the initial state.
  state = deepcopy(init)

  # Checking number of monitors.
  num_monitors = length(monitors)
  println("Number of monitors: $(num_monitors)"); flush(stdout)
  @assert num_monitors == length(nsamps)

  # Total number of MCMC iterations
  nmcmc = nburn + thin * nsamps[1] 
  verbose > 0 && println("Total number of MCMC iterations: $(nmcmc)")

  # Thin factor for each monitor
  thins = div.(nmcmc - nburn, nsamps)
  println("Thinning factors: ", thins)

  # Check monitor
  if num_monitors == 0
    verbose > 0 && println("Using default monitor.")
    fnames = fieldnames(typeof(init))
    append!(monitors, [fnames])
    num_monitors = 1
  end

  verbose > 0 && println("Preallocating memory...")

  # Create object to return
  @time chain = [fill(deepcopyFields(state, monitors[i]), nsamps[i])
                 for i in 1:num_monitors]

  # Initialize empty summary statistics.
  summary_stats = []

  # Initialize sample index tracker for each monitor.
  counters = zeros(Int, num_monitors)

  # Initialize progress bar.
  pbar = ProgressBar(1:nmcmc)

  # Print time at beginning of MCMC.
  println("Start time: ", showtime()); flush(stdout)

  # Gibbs loop.
  for i in pbar
    update!(state)
    if i > nburn
      for j in 1:num_monitors
        if i % thins[j] == 0
          substate = deepcopyFields(state, monitors[j])
          counters[j] += 1
          chain[j][counters[j]] = substate
        end
      end
    end

    # Execute callback function.
    summary_stat = callback_fn(state, i, pbar)

    # Append summary statistics if callback returns something.
    if summary_stat != nothing
      append!(summary_stats, [summary_stat])
    end

    # This is needed so that `/usr/bin/tail` in works properly
    # in a BASH terminal.
    flush(stdout)
  end

  # Print end time.
  println("End time: ", showtime()); flush(stdout)

  # Return chain, last state, and summary statistics.
  return (chain, state, summary_stats)
end
