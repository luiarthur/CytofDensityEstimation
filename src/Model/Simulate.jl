"""
Computes the parameters of the inverse gamma distribution, given a mean and
standard deviation.
"""
function invgammamoment(m, s)
  v = s ^ 2
  a = (m / s) ^ 2 + 2
  b = m * (a - 1)
  return a, b
end


function generate_sample(; N::Int, gamma::Real, eta::AbstractVector{<:Real},
                         loc::AbstractVector{<:Real},
                         scale::AbstractVector{<:Real},
                         df::AbstractVector{<:Real},
                         skew::AbstractVector{<:Real})
  K = length(eta)
  Ninf= rand(Binomial(N, gamma))
  Nfinite = N - Ninf
  lam = rand(Categorical(eta), Nfinite)
  yfinite = Util.randskewt.(loc[lam], scale[lam], df[lam], skew[lam])
  return [yfinite; fill(-Inf, Ninf)]
end


function generate_samples(; NC, NT, gammaC, gammaT, etaC, etaT, 
                          loc, scale, df, skew)
  K = length(loc)
  @assert K == length(etaC) == length(etaT)
  @assert K == length(scale) == length(df) == length(skew)
  yC = generate_sample(N=NC, gamma=gammaC, eta=etaC, loc=loc, scale=scale,
                       df=df, skew=skew)
  yT = generate_sample(N=NT, gamma=gammaT, eta=etaT, loc=loc, scale=scale,
                       df=df, skew=skew)
  return (yC=yC, yT=yT, gammaC=gammaC, gammaT=gammaT, etaC=etaC, etaT=etaT,
          loc=loc, scale=scale, df=df, skew=skew)
end
