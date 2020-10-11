"""
Partitions a list `xs` into `(good, bad)` according
to a condition.
"""
function partition(condition::Function, xs::Vector{T}) where T
  good = T[]
  bad = T[]

  for x in xs
    if condition(x)
      append!(good, [x])
    else
      append!(bad, [x])
    end
  end

  return good, bad
end


