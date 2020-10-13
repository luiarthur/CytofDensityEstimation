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
  # TODO
end
