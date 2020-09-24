library(truncnorm)

skew_t = function(n, nu, loc, scale, skew) {
  w = rgamma(n, nu/2, nu/2)
  z = rtruncnorm(n, 0, Inf, 0, 1/w)
  rnorm(n, loc + z * skew, scale/w)
}

x = skew_t(1e6, 30, 3, .2, -1)
hist(x)
