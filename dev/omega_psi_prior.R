# NOTE: Need to pick prior for (psi, omega) carefully.

tophi = function(psi, omega) psi / sqrt(omega)
tosig = function(psi, omega) sqrt(psi^2 + omega)
topsi = function(phi, sigma) phi * sqrt(toomega(phi, sigma))
toomega = function(phi, sigma) sigma^2 / (1 + phi^2)


n = 1e5

# omega = (1 / rgamma(n, 2, 1e-3))
b = rgamma(n, .5, 1)
omega = 1 / rgamma(n, 2.5, b)
# omega = 1 / rgamma(n, .1, .1)
psi = rnorm(n, -1, 1)

phi = tophi(psi, omega)
sig = tosig(psi, omega)
plot(phi, sig, pch=20, cex=.1, xlim=c(-20, 20), ylim=c(0,4))
# mean(sig < 1 & phi < -5)
# plot(psi, omega, cex=.1, ylim=c(0, 0.1))  # needs to be uniform between (0, 0.02) for omega

# toomega(-10, c(.1, 2))
# topsi(-10, c(.1, 2))

# Make priors so that I get these (sig, phi).
# n=1e4
# sig = runif(n, 0, 2)
# phi = runif(n, -20, 20)
# # sig = runif(n, .5, .6)
# # phi = runif(n, -20, 0)
# plot(phi, sig, cex=.1, pch=20)
# omega = toomega(phi=phi, sig=sig)
# psi = topsi(phi=phi, sig=sig)
# plot(psi, omega, pch=20, cex=.5, ylim=c(0, .1))
