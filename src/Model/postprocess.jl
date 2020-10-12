function group(sym::Symbol, chain)
  for monitor in chain
    if sym in keys(monitor[1])
      out = [s[sym] for s in monitor]
      return out
    end
  end
end
