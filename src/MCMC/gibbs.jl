# https://github.com/cloud-oak/ProgressBars.jl
# set_description(iter, string(@sprintf("Loss: %.2f", loss)))
import Dates
using ProgressBars
include("partition.jl")

monitor_default() = Vector{Vector{Symbol}}([])
thin_default() = Int[]

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
               monitors::Vector{Vector{Symbol}}=monitor_default(),
               thins::Vector{Int}=thin_default(),
               nmcmc::Int=1000, nburn::Int=0,
               loglike=[], verbose::Int=1) where T

  # Make a copy of the initial state.
  state = deepcopy(init)

  # Checking number of monitors.
  numMonitors = length(monitors)
  println("Number of monitors: $(numMonitors)"); flush(stdout)
  @assert numMonitors == length(thins)

  # Check monitor
  if numMonitors == 0
    println("Using default monitor."); flush(stdout)
    fnames = [fname for fname in fieldnames(typeof(init))]
    append!(monitors, [fnames])
    append!(thins, 1)
    numMonitors = 1
  end

  # Number of Samples for each Monitor
  numSamps = [div(nmcmc, thins[i]) for i in 1:numMonitors]

  verbose <= 0 || (println("Preallocating memory..."); flush(stdout))

  # Create object to return
  @time out = [fill(deepcopyFields(state, monitors[i]), numSamps[i])
               for i in 1:numMonitors]

  function Msg(i::Int)
    if (printFreq > 0) && (i % printFreq == 0) && (i > 1)
      loglikeMsg = ismissing(loglike) ? "" : "-- loglike: $(last(loglike))"
      print("$(showtime()) -- $(i)/$(nburn+nmcmc) $loglikeMsg"); flush(stdout)

      if printlnAfterMsg
        println(); flush(stdout)
      end

      flush(stdout)
    end
  end

  counters = zeros(Int, numMonitors)

  println(showtime()); flush(stdout)
  pbar = ProgressBar(1:(nburn + nmcmc))

  # Gibbs loop.
  for i in pbar
    update!(state, i)
    if i > nburn
      for j in 1:numMonitors
        if i % thins[j] == 0
          substate = deepcopyFields(state, monitors[j])
          counters[j] += 1
          out[j][counters[j]] = substate
        end
      end
    end
    if length(loglike) > 0
      ll = round(last(loglike), digits=2)
      set_description(pbar, "loglike: $(ll)")
    end
    flush(stdout)
  end

  println(showtime()); flush(stdout)
  return (out, state)
end
