library(truncnorm)

dskew= function(x, nu, loc, scale, alpha, log=F) {
  z = (x - loc) / scale
  u = alpha * z * sqrt((nu + 1) / (nu + z*z))
  kernel = dt(z, nu, log=T) + pt(u, nu + 1, log=T)
  out = kernel + log(2) - log(scale)
  if (log==TRUE) return(out) else return(exp(out))
}

rskewt = function(n, nu, xi, psi, omega) {
  w = rgamma(n, nu/2, nu/2)
  z = rtruncnorm(n, 0, Inf, 0, sqrt(1/w))
  xi + psi * z + rnorm(n) * sqrt(omega/w)
}

# Reparameterized parameters.
nu=runif(1, 3, 30)
xi=rnorm(1, 0, 2)
psi=rnorm(1, -1, 2)
omega=rgamma(1, 3, 3)

# Transform parameters.
alpha = psi/sqrt(omega)
sigma = sqrt(omega + psi^2)
xx = seq(-15, 6, len=1000)

hist(rskewt(1e6, nu=nu, xi=xi, psi=psi, omega=omega), breaks=300, freq=F)
lines(xx, dskew(xx, nu, xi, sigma, alpha), lwd=3, col='blue')

cat('nu:', nu, 'xi:', xi, 'psi:', psi, 'omega:', omega,
    'alpha:', alpha, 'sigma:', sigma, '\n')
