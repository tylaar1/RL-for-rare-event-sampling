struct ExcursionProblem
    rewards::Array{Float64,3}
    trajectory_length::Int
    γ::Float64
end

struct ExcursionStateSpace
    problem::ExcursionProblem
end

struct Trajectory
    states::Vector{Tuple{Int64,Int64,Int64}}
    rewards::Vector{Float64}
    actions::Vector{Int64}
end

struct PolicyGradient
    discount:: Float64
    _policy_parameters::Dict{Tuple{Int64,Int64,Int64},Float64}
    _parameter_gradients::Dict{Tuple{Int64,Int64,Int64},Float64}
end

struct ExactSolution
    values::Dict{Tuple{Int64,Int64,Int64},Float64}
    policy::Dict{Tuple{Int64,Int64,Int64},Float64}
end

struct ExcursionProblem3D
    rewards::Array{Float64,3}
    trajectory_lengths::Array{Int64}
    γ::Float64
end

struct ExcursionStateSpace3D
    problem::ExcursionProblem3D
    trajectory_length::Int
end
