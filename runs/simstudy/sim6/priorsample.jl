# TEST:
function priorsample(y=range(-8, 4, length=200))
  tau = rand(Gamma(.5, 1))
  omega = rand(InverseGamma(2.5, tau))
  psi = rand(Normal(-1, .5))
  mu = rand(Normal(1, 1))
  sigma = CDE.Util.scalefromaltskewt(sqrt(omega), psi)
  phi = CDE.Util.skewfromaltskewt(sqrt(omega), psi)
  nu = rand(LogNormal(3, .5))
  println("(tau,omega,psi): $((tau,omega,psi))")
  println("(sigma,phi): $((sigma,phi))")
  return y, pdf.(CDE.Util.SkewT(mu, sigma, nu, phi), y)
end
dopriorsample() = let
  _y, ypdf = priorsample()
  plot(_y, ypdf, label=nothing); savefig("results/tmp.pdf"); closeall()
end
