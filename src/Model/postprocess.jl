default_ygrid(; lo=-6, hi=6, length=1000) = range(lo, hi, length=length)

function group(sym::Symbol, chain; altskew_sym=:psi, altvar_sym=:omega)
  if sym in (:sigma, :phi)
    scale, skew = fetch_skewt_stats(chain; altskew_sym=altskew_sym,
                                    altvar_sym=altvar_sym)
    return sym == :sigma ? scale : skew
  else
    for monitor in chain
      if sym in keys(monitor[1])
        return [s[sym] for s in monitor]
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

  return (scale=scale, skew=skew)
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


function printsummary(chain, summarystats; laststate=nothing, digits=3)
  ll = [s[:loglike] for s in summarystats]
  p = group(:p, chain)
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

  sanitize(s) = round(mean(s), digits=digits)
  sanitizevec(v) = round.(mean(v), digits=digits)

  println("mean p: ", sanitize(p))
  println("mean beta: ", sanitize(beta))
  println("mean gammaC: ", sanitize(gammaC))
  println("mean gammaT: ", sanitize(gammaT))
  any(beta) && println("mean gammaT[β=1]: ", sanitize(gammaT[beta .== 1]))
  all(beta) || println("mean gammaT[β=0]: ", sanitize(gammaT[beta .== 0]))
  println("mean etaC: ", sanitizevec(etaC))
  println("mean etaT: ", sanitizevec(etaT))
  any(beta) && println("mean etaT[β=1]: ", sanitizevec(etaT[beta .== 1]))
  all(beta) || println("mean etaT[β=0]: ", sanitizevec(etaT[beta .== 0]))
  println("mean mu: ", sanitizevec(mu))
  println("mean sigma: ", sanitizevec(sigma))
  println("mean nu: ", sanitizevec(nu))
  println("mean phi: ", sanitizevec(phi))

  if laststate != nothing
  end
end


function plot_posterior_predictive(yC, yT, chain, bw; lw=3, labelyC=L"y_C",
                                   ygrid=default_ygrid(),
                                   labelyT=L"y_T", legendfontsize=12,
                                   showT=true, alpha=0.3,
                                   labelCppd="post. density",
                                   labelTppd="post. density", 
                                   simdata=nothing, density_legend_pos=:best)

  pdfC, pdfT = posterior_density(chain, ygrid)
  beta = group(:beta, chain)

  pdfC_lower = MCMC.quantiles(hcat(pdfC...), .025, dims=2)
  pdfC_upper = MCMC.quantiles(hcat(pdfC...), .975, dims=2)
  pdfC_mean = mean(pdfC)

  if any(beta)
    pdfT_lower = MCMC.quantiles(hcat(pdfT...), .025, dims=2)
    pdfT_upper = MCMC.quantiles(hcat(pdfT...), .975, dims=2)
    pdfT_mean = mean(pdfT)
  end

  # Plot density
  legendfont = font(12)

  if simdata == nothing
    plot(kde(yC[isfinite.(yC)], bandwidth=bw), lw=3, label=labelyC,
         # legendfont=legendfont,
         color=:blue, ls=:dot)
  else
    plot(ygrid, pdf.(simdata[:mmC], ygrid), lw=3, label=labelyC,
         color=:blue, ls=:dot)
  end
  plot!(ygrid, pdfC_lower, fillrange=pdfC_upper, alpha=alpha, color=:blue,
        label=labelCppd, legend=density_legend_pos)

  if simdata == nothing
    plot!(kde(yT[isfinite.(yT)], bandwidth=bw), lw=3, label=labelyT,
          color=:red, ls=:dot)
  else
    plot!(ygrid, pdf.(simdata[:mmT], ygrid), lw=3, label=labelyT,
          color=:red, ls=:dot)
  end
  if any(beta)
    plot!(ygrid, pdfT_lower, fillrange=pdfT_upper, alpha=alpha, color=:red,
          label=labelTppd)
  end
end


function plot_gamma(yC, yT, chain)
  gammaC = group(:gammaC, chain)
  gammaT = group(:gammaT, chain)
  beta = group(:beta, chain)

  gammaT_star = any(beta) ? gammaT[beta] : gammaC

  boxplot(gammaC, outliers=false, color=:blue, label="", alpha=.5)
  boxplot!(gammaT_star, outliers=false, color=:red, label="", alpha=.5)
  xticks!([1,2], [L"\gamma_C", L"\gamma_T^\star"],
          xtickfont=font(20), ytickfont=font(16))
  scatter!([1, 2], [mean(isinf.(yC)), mean(isinf.(yT))], markersize=[10, 10], 
           color=[:blue, :red], label=nothing)
end


function boxplot_kernel_param(sym, chain; paramname="", simdata=nothing,
                              simsym=nothing)
  param = group(sym, chain)
  boxplot(hcat(param...)', outliers=false, labels=nothing)
  xlabel!("$(paramname) Mixture Components", font=font(12))
  simdata == nothing || hline!(simdata[simsym], ls=:dot, lw=3, labels="truth")
  return
end


function trace_kernel_param(sym, chain; paramname="", simdata=nothing,
                            simsym=nothing)
  param = group(sym, chain)
  plot(hcat(param...)', outliers=false, labels=nothing)
  xlabel!("MCMC iteration", font=font(12))
  ylabel!(paramname, font=font(12))
  simdata == nothing || hline!(simdata[simsym], ls=:dot, labels="truth")
  return
end


function plotpostsummary(chain, summarystats, yC, yT, imgdir; digits=3,
                         laststate=nothing, bw_postpred=0.3, simdata=nothing,
                         ygrid=default_ygrid(), xlims_=nothing,
                         plotsize=(400,400), density_legend_pos=:best)
  # Loglikelihood trace
  ll = [s[:loglike] for s in summarystats]
  plot(ll, label=nothing)
  ylabel!("log likelihood")
  xlabel!("MCMC iteration")
  plot!(size=plotsize)
  savefig("$(imgdir)/loglike.pdf")
  closeall()

  # Posterior density
  plot_posterior_predictive(yC, yT, chain, bw_postpred, ygrid=ygrid,
                            density_legend_pos=density_legend_pos)
  xlims_ == nothing || xlims!(xlims_)
  plot!(size=plotsize)
  savefig("$(imgdir)/postpred.pdf")
  closeall()

  if simdata != nothing
    plot_posterior_predictive(yC, yT, chain, bw_postpred, ygrid=ygrid,
                              simdata=simdata,
                              density_legend_pos=density_legend_pos)
    xlims_ == nothing || xlims!(xlims_)
    plot!(size=plotsize)
    savefig("$(imgdir)/postpred-true-data-density.pdf")
    closeall()
  end

  # Trace of p, beta.
  trace_kernel_param(:p, chain, paramname="p")
  plot!(size=plotsize)
  savefig("$(imgdir)/p-trace.pdf"); closeall()
  trace_kernel_param(:beta, chain, paramname="β")
  plot!(size=plotsize)
  savefig("$(imgdir)/beta-trace.pdf"); closeall()

  # Proportion of gammas.
  plot_gamma(yC, yT, chain)
  plot!(size=plotsize)
  savefig("$(imgdir)/gammas.pdf")
  closeall()

  # Plot kernel parameters boxplot
  boxplot_kernel_param(:mu, chain, paramname="μ", simdata=simdata, simsym=:loc)
  plot!(size=plotsize)
  savefig("$(imgdir)/mu.pdf"); closeall()

  boxplot_kernel_param(:sigma, chain, paramname="σ", simdata=simdata, simsym=:scale)
  plot!(size=plotsize)
  savefig("$(imgdir)/sigma.pdf"); closeall()

  boxplot_kernel_param(:nu, chain, paramname="ν", simdata=simdata, simsym=:df)
  plot!(size=plotsize)
  savefig("$(imgdir)/nu.pdf"); closeall()

  boxplot_kernel_param(:phi, chain, paramname="ϕ", simdata=simdata, simsym=:skew)
  plot!(size=plotsize)
  savefig("$(imgdir)/phi.pdf"); closeall()

  boxplot_kernel_param(:psi, chain, paramname="ψ")
  plot!(size=plotsize)
  savefig("$(imgdir)/psi.pdf"); closeall()

  boxplot_kernel_param(:omega, chain, paramname="ω")
  plot!(size=plotsize)
  savefig("$(imgdir)/omega.pdf"); closeall()

  boxplot_kernel_param(:etaC, chain, paramname="ηc", simdata=simdata, simsym=:etaC)
  plot!(size=plotsize)
  savefig("$(imgdir)/etaC.pdf"); closeall()

  boxplot_kernel_param(:etaT, chain, paramname="ηt", simdata=simdata, simsym=:etaT)
  plot!(size=plotsize)
  savefig("$(imgdir)/etaT.pdf"); closeall()

  # Plot kernel parameters trace plot
  trace_kernel_param(:mu, chain, paramname="μ")
  plot!(size=plotsize)
  savefig("$(imgdir)/mu-trace.pdf"); closeall()

  trace_kernel_param(:sigma, chain, paramname="σ")
  plot!(size=plotsize)
  savefig("$(imgdir)/sigma-trace.pdf"); closeall()

  trace_kernel_param(:nu, chain, paramname="ν")
  plot!(size=plotsize)
  savefig("$(imgdir)/nu-trace.pdf"); closeall()

  trace_kernel_param(:phi, chain, paramname="ϕ")
  plot!(size=plotsize)
  savefig("$(imgdir)/phi-trace.pdf"); closeall()

  trace_kernel_param(:etaC, chain, paramname="ηc")
  plot!(size=plotsize)
  savefig("$(imgdir)/etaC-trace.pdf"); closeall()

  trace_kernel_param(:etaT, chain, paramname="ηt")
  plot!(size=plotsize)
  savefig("$(imgdir)/etaT-trace.pdf"); closeall()

  trace_kernel_param(:psi, chain, paramname="ψ")
  plot!(size=plotsize)
  savefig("$(imgdir)/psi-trace.pdf"); closeall()

  trace_kernel_param(:omega, chain, paramname="ω")
  plot!(size=plotsize)
  savefig("$(imgdir)/omega-trace.pdf"); closeall()
end
