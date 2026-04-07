function reward(problem::ExcursionProblem, s′)
    x′, t′ = s′
    problem.rewards[x′ + problem.trajectory_length + 1, t′]
end
function next_state(::ExcursionProblem, s, a)
    x, t = s
    (Int64(x + 2a - 3), t+1) #for some reason is converted to float 32 in NN
end
function is_terminal(problem::ExcursionProblem, s)
    _, t = s
    return t == problem.trajectory_length
end
function starting_state(problem::ExcursionProblem)
    return (0, 0)
end

function Trajectory()
    Trajectory([], Float64[], Int64[])
end

function partial_returns(traj::Trajectory,discount::Float64)
    returns = Float64[]
    cumulative_return = 0.0
    for r in Iterators.reverse(traj.rewards)
        cumulative_return = r + (discount*cumulative_return)
        pushfirst!(returns, cumulative_return)
    end
    return returns 
end

function n_transitions(traj::Trajectory)
    @assert length(traj.actions) == length(traj.rewards)
    @assert length(traj.actions) == length(traj.states) - 1
    return length(traj.actions)
end

function append_transition(traj::Trajectory, next_state::Tuple{Int64,Int64}, action::Number, reward::Float64)
    push!(traj.states, next_state)
    push!(traj.actions, action)
    push!(traj.rewards, reward)
end

function transitions(traj::Trajectory, discount::Float64)
    T = n_transitions(traj)
    out = Vector{Tuple{Tuple{Int64,Int64},Int64,Tuple{Int64,Int64},Float64,Float64}}(undef, T)
    return_to_go = 0.0
    for t in T:-1:1 #T:-1:1 does reverse
        return_to_go = traj.rewards[t] + (discount*return_to_go)
        out[t] = (traj.states[t], traj.actions[t], traj.states[t+1], traj.rewards[t], return_to_go)
    end
    return out
end

function def_problem(T::Int64,bias::Float64,negative_penalty::Float64)
    #R = Random.randn(Float64, 2T+1, T)
    R = zeros( 2T+1, T) #now that we have arbitrary T we want to avoid learning noise
    R[1:T, :] .+= negative_penalty
    R[:, T] .= (-T:T) .^ 2 .* (-bias)
    return R
end