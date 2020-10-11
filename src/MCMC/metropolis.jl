function metropolisBase(curr::Real, log_prob::Function, stepSD::Real)
  cand = curr + randn() * stepSD
  logU = log(rand())
  logP = log_prob(cand) - log_prob(curr)
  accept = logP > logU
  draw = accept ? cand : curr
  return (draw, accept)
end


function metropolis(curr::Real, log_prob::Function, stepSD::Real)
  return metropolisBase(curr, log_prob, stepSD)[1]
end


function metropolis(curr::AbstractVector, log_prob::Function,
                    stepSD::AbstractMatrix)
  cand = rand(MvNormal(curr, stepSD))
  logU = log(rand())
  p = log_prob(cand) - log_prob(curr)
  out = p > logU ? cand : curr
  return out
end


"""
Adaptive metropolis (within Gibbs). See section 3 of the paper below:
  http://probability.ca/jeff/ftpdir/adaptex.pdf

Another useful website:
  https://m-clark.github.io/docs/ld_mcmc/index_onepage.html
"""
function metropolisAdaptive(curr::Real, log_prob::Real,
                            tuner::TuningParam;
                            update::Function=update_tuning_param_default)
  draw, accept = metropolisBase(curr, log_prob, tuner.value)
  update(tuner, accept)
  return draw
end


"""
Log absolute jacobian term to be added when taking the log of a postiviely
supported random variable (x), so as to produce a random variable with
support on the real line. When added to the target log density, the resulting
expression can be used in a metropolis step conveniently.
"""
function logabsjacobian_logx(real_x::Real)
  return real_x
end


"""
Log absolute jacobian term to be added when taking the logit of a
lower-and-upper-bounded random variable (x), so as to produce a random variable
with support on the real line. When added to the target log density, the
resulting expression can be used in a metropolis step conveniently.

# Arguments
- `real_x::Float64`: the unconstrained value of the positively-supported
                     parameter x (i.e. log x) 
- `a::Float64`: lower bound for unconstrained parameter (default: 0)
- `b::Float64`: upper bound for unconstrained parameter (default: 1)
"""
function logabsjacobian_logitx(real_x::Real)
  return logpdf(Logistic(), real_x)
end


function metLogAdaptive(curr::Real, log_prob::Function, tuner::TuningParam;
                        update::Function=update_tuning_param_default)

  function log_prob_plus_logabsjacobian(log_x::Real)
    x = exp(log_x)
    return log_prob(x) + logabsjacobian_logx(log_x)
  end

  log_x = metropolisAdaptive(log(curr), log_prob_plus_logabsjacobian, tuner,
                             update=update)

  return exp(log_x)
end


function metLogitAdaptive(curr::Real, log_prob::Function,
                          tuner::TuningParam;
                          update::Function=update_tuning_param_default)

  function log_prob_plus_logabsjacobian(logit_x::Real)
    x = logistic(logit_x)
    return log_prob(x) + logabsjacobian_logitx(logit_x)
  end

  logit_x = metropolisAdaptive(logit(curr),
                               log_prob_plus_logabsjacobian,
                               tuner, update=update)

  return logistic(logit_x)
end


"""
weighted sampling: takes (unnormalized) log probs and returns index
"""
function wsample_logprob(logProbs::Vector{T}) where {T <: Number}
  log_p_max = maximum(logProbs)
  p = exp.(logProbs .- log_p_max)
  return Distributions.wsample(p)
end


"""
log 1 minus. For example, `log1m(.3) == log(1 - .3) == log1p(-.3) == log(.7)`.
"""
log1m(x::T) where {T <: Number} = log1p(-x)


function to_onehot(indvec::Vector{Int}, categories::Int)
  N = length(indvec)
  onehot_matrix = zeros(Int, N, categories)
  for n in 1:N
    onehot_matrix[n, indvec[n]] = 1
  end
  return onehot_matrix
end
