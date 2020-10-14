function group(sym::Symbol, chain; altskew_sym=:psi, altvar_sym=:omega)
  if sym in (:sigma, :phi)
    skew, scale = fetch_skewt_stats(chain; altskew_sym=:psi, altvar_sym=:omega)
    return sym == :sigma ? scale : skew
  else
    for monitor in chain
      if sym in keys(monitor[1])
        out = [s[sym] for s in monitor]
        return out
      end
    end
  end
end


function fetch_skewt_stats(chain; altskew_sym=:psi, altvar_sym=:omega)
  altskew = group(altskew_sym, chain)
  altvar = group(altvar_sym, chain)

  skew = [Util.skewfromaltskewt.(sqrt.(altvar[b]), altskew[b])
          for b in eachindex(altskew)]
  scale = [Util.scalefromaltskewt.(sqrt.(altvar[b]), altskew[b])
           for b in eachindex(altskew)]

  return (skew=skew, scale=scale)
end


function posterior_density(chain, ygrid)
  mu = group(:mu, chain)
  sigma = group(:sigma, chain)
  nu = group(:nu, chain)
  phi = group(:phi, chain)
  etaC = group(:etaC, chain)
  etaT = group(:etaT, chain)
  beta = group(:beta, chain)
  K = length(mu[1])
  B = length(beta)

  function compute_pdf(b, eta)
    comps = [SkewT(mu[b][k], sigma[b][k], nu[b][k], phi[b][k]) for k in 1:K]
    pdf.(MixtureModel(comps , eta[b]), ygrid)
  end

  pdfC = [compute_pdf(b, etaC) for b in 1:B]

  if any(beta)
    pdfT = [compute_pdf(b, etaT) for b in 1:B if beta[b]]
  else
    pdfT = nothing
  end

  return (pdfC=pdfC, pdfT=pdfT)
end


function printsummary(chain, summarystats, laststate=nothing)
  ll = [s[:loglike] for s in summarystats]
  gammaC = group(:gammaC, chain)
  gammaT = group(:gammaT, chain)
  phi = group(:phi, chain)
  etaC = group(:etaC, chain)
  etaT = group(:etaT, chain)
  mu = group(:mu, chain)
  beta = group(:beta, chain)
  sigma = group(:sigma, chain)
  phi = group(:phi, chain)
  nu = group(:nu, chain)
  psi = group(:psi, chain)
  omega = group(:omega, chain)

  println("mean beta: ", mean(beta))
  println("mean gammaC: ", mean(gammaC))
  println("mean gammaT[β=1]: ", mean(gammaT[beta .== 1]))
  println("mean etaC: ", round.(mean(etaC), digits=3))
  println("mean etaT[β=1]: ", round.(mean(etaT[beta .== 1]), digits=3))
  println("mean mu: ", round.(mean(mu), digits=3))
  println("mean sigma: ", round.(mean(sigma), digits=3))
  println("mean nu: ", round.(mean(nu), digits=3))
  println("mean phi: ", round.(mean(phi), digits=3))

  if laststate != nothing
  end
end
