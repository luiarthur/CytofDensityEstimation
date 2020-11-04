mutable struct MvTuner{F <: Real}
  current_mean::Vector{F}
  current_cov::Matrix{F}
  beta::F
  iter::Int
  d::Int
end


MvTuner(d::Integer, T::Type=Float64) = MvTuner(zeros(T, d), eye(T, d)*0.01/d,
                                               T(.05), 1, d)


function update!(x::AbstractVector{<:Real}, tuner::MvTuner{<:Real})
  tuner.iter += 1
  update_mean!(tuner.current_mean, x, tuner.iter)
  update_cov!(tuner.current_cov, tuner.current_mean, x, tuner.iter)
end
