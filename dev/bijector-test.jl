import Pkg; Pkg.activate("../")
using Turing: Bijectors

b = Bijectors.SimplexBijector()
ib = inv(b)
x = [.3, .5, .2]
y = b(x)

Bijectors.logabsdetjac(b, x)
Bijectors.logabsdetjac(ib, y)
