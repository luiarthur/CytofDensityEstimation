library(truncnorm)

d_skew_t = function(x, nu, loc, scale, alpha) {
  z = (x - loc) / scale
  u = alpha * z * sqrt((nu + 1) / (nu + z*z))
  kernel = dt(z, nu, log=T) + pt(u, nu + 1, log=T)
  kernel + log(2) - log(scale)
}

skew_t = function(n, nu, loc, scale, alpha) {
  w = rgamma(n, nu/2, nu/2)
  z = rtruncnorm(n, 0, Inf, 0, sqrt(1/w))
  # rnorm(n, loc + z * alpha, scale/sqrt(w))

  delta = alpha / sqrt(1 + alpha^2)
  loc + scale * z * delta + scale * sqrt(1 - delta^2) * rnorm(n)
}


nu=30
loc=3
scale=.2
alpha=-10

x = skew_t(1e6, nu, loc, scale, alpha)
xx = seq(-10, 10, len=10000)
hist(x, prob=T, breaks=100)
lines(xx, exp(d_skew_t(xx, nu, loc, scale, alpha)), lwd=3, col='blue')
