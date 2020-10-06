module Helper
using SpecialFunctions

std_t_lcdf(x, nu) = log(std_t_cdf(x, nu))
function std_t_cdf(x, df)
  x_t = df / (x^2 + df) 
  neg_cdf = 0.5 * beta_inc(0.5 * df, 0.5, x_t)[1]
  return x < 0 ? neg_cdf : 1 - neg_cdf
end

end
