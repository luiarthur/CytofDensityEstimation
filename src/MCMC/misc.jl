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


"""
x: real vector of dim K - 1
return: simplex of dim K
"""
function simplex_invtransform(x::AbstractVector{<:Real})
  K = length(x) + 1
  z = logistic.(x - log.(K .- collect(1:K-1)))
  p = zeros(K)
  for k in 1:K-1
    p[k] = (1 - sum(p[1:k-1])) * z[k]
  end
  p[K] = 1 - sum(p[1:K-1])
  return p  # simplex
end


"""
Given simplex p (length K), return real vector of length K-1.
"""
function simplex_transform(p::AbstractVector{<:Real})
  K = length(p)
  ks = collect(1:K-1)
  z = [p[k] / (1 - sum(p[1:k-1])) for k in ks]
  return logit.(z) + log.(K .- ks)
end


"""
x: real vector of dim K - 1
return: log abs value of determinant of x
"""
function simplex_logabsdet(x::AbstractVector{<:Real})
  K = length(x) + 1
  ks = collect(1:K-1)
  z = logistic.(x - log.(K .- ks))
  p = simplex_invtransform(x)
  return sum(log1p.(-z) + log.(p[ks]))
end

# DIC.
deviance(loglikelihood::Real) = -2 * loglikelihood
function dic(loglikelihood::AbstractVector{<:Real})
  D = deviance.(loglikelihood)
  return mean(D) + 0.5 * var(D)
end
