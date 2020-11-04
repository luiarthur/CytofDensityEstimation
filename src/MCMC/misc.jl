eye(T::Type, n::Integer) = Matrix{T}(LinearAlgebra.I(n))
eye(n::Integer) = eye(Float64, n)


"""
weighted sampling: takes (unnormalized) log probs and returns index
"""
function wsample_logprob(logProbs::AbstractVector{<:Real})
  log_p_max = maximum(logProbs)
  p = exp.(logProbs .- log_p_max)
  return Distributions.wsample(p)
end


"""
log 1 minus. For example, `log1m(.3) == log(1 - .3) == log1p(-.3) == log(.7)`.
"""
log1m(x::T) where {T <: Real} = log1p(-x)


"""
Convert a vector of indices to a one-hot matrix, given also the number of
categories.
"""
function to_onehot(indvec::Vector{Int}, categories::Int)
  N = length(indvec)
  onehot_matrix = zeros(Int, N, categories)
  for n in 1:N
    onehot_matrix[n, indvec[n]] = 1
  end
  return onehot_matrix
end


"""
log pdf of gaussian mixture model, for a vector of x of size N and (m: mixture
locations, s: mixture scales, w: mixture weights) each vectors of size K
"""
function gmmlogpdf(m, s, w, x; dims)
  return logsumexp(normlogpdf.(m, s, x) .+ log.(w), dims=dims)
end


"""
log pdf of gaussian mixture model, for a scalar x N and (m: mixture locations,
s: mixture scales, w: mixture weights) each vectors of size K
"""
function gmmlogpdf(m, s, w, x::Real)
  return logsumexp(normlogpdf.(m, s, x) .+ log.(w))
end


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


function quantiles(X, q; dims, drop=false)
  Q = mapslices(x -> quantile(x, q), X, dims=dims)
  out = drop ? dropdims(Q, dims=dims) : Q
  return out
end


function update_mean!(current_mean, x, iter)
  current_mean .+= (x - current_mean) / iter
end


function update_cov!(current_cov, current_mean, x, iter)
  d = x - current_mean
  current_cov .= current_cov * (iter - 1)/iter + (d*d') * (iter - 1)/iter^2
  current_cov .= Matrix(LinearAlgebra.Symmetric(current_cov))
end


#= Test
using Statistics
n = 300000
x = [randn(4) for i in 1:n]
current_mean = mean(x[1:2])
current_cov = cov(x[1:2])
for i in 2:n
  update_mean!(current_mean, x[i], i) 
  update_cov!(current_cov, current_mean, x[i], i) 
end
cov(x)
current_cov
current_mean, mean(x)
=#
