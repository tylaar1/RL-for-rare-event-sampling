function reward(problem::ExcursionProblem, s′)
    x′, t′,T = s′
    problem.rewards[x′ + T + 1, t′,T]
end

function reward(problem::ExcursionProblem3D, s′)
    x′, t′,T = s′
    problem.rewards[x′ + T + 1, t′,T]
end
function next_state(::ExcursionProblem, s, a)
    x, t ,T = s
    (Int64(x + 2a - 3), t+1,T) #for some reason is converted to float 32 in NN
end

function next_state(::ExcursionProblem3D, s, a)
    x, t ,T = s
    (Int64(x + 2a - 3), t+1,T) #for some reason is converted to float 32 in NN
end
function is_terminal(problem::ExcursionProblem, s)
    _, t, T = s
    return t == T
end

function is_terminal(problem::ExcursionProblem3D, s)
    _, t, T = s
    return t == T
end


function starting_state(problem::ExcursionProblem)
    T = problem.trajectory_length
    return (0, 0,T)
end

function starting_state(problem::ExcursionProblem3D,T)
    return (0, 0,T)
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

function append_transition(traj::Trajectory, next_state::Tuple{Int64,Int64,Int64}, action::Number, reward::Float64)
    push!(traj.states, next_state)
    push!(traj.actions, action)
    push!(traj.rewards, reward)
end

function transitions(traj::Trajectory, discount::Float64)
    T = n_transitions(traj)
    out = Vector{Tuple{Tuple{Int64,Int64,Int64},Int64,Tuple{Int64,Int64,Int64},Float64,Float64}}(undef, T)
    return_to_go = 0.0
    for t in T:-1:1 #T:-1:1 does reverse
        return_to_go = traj.rewards[t] + (discount*return_to_go)
        out[t] = (traj.states[t], traj.actions[t], traj.states[t+1], traj.rewards[t], return_to_go)
    end
    return out
end

function def_problem(T::Int64,bias::Float64,negative_penalty::Float64)
    #R = Random.randn(Float64, 2T+1, T)
    R = zeros( 2T+1, T,T) #now that we have arbitrary T we want to avoid learning noise
    R[1:T, :,T] .+= negative_penalty
    R[:, T,T] .= (-T:T) .^ 2 .* (-bias)
    return R
end

function def_3D_problem(T_min::Int64,T_max::Int64,bias::Float64,negative_penalty::Float64)
    R = zeros( 2*T_max+1, T_max, T_max)
    for T in T_min:T_max
        if T%2 == 0
            R[1:T, :,T] .+= negative_penalty
            R[1:2T+1, T,T] .= (-T:T) .^ 2 .* (-bias) 
        end
    end
    return R
end