using CairoMakie
using ProgressBars
using Random

# Training struct
struct ExcursionProblem
    rewards::Matrix{Float64}
    trajectory_length::Int
end

function reward(problem::ExcursionProblem, s, a, s′)
    x′, t′ = s′
    problem.rewards[x′ + problem.trajectory_length + 1, t′]
end
function next_state(::ExcursionProblem, s, a)
    x, t = s
    (x + 2a - 3, t+1)
end
function is_terminal(problem::ExcursionProblem, s)
    _, t = s
    return t == problem.trajectory_length
end
function action_space(problem::ExcursionProblem, s)
    return 1:2 # tuple of actions
end
function starting_state(problem::ExcursionProblem)
    return (0, 0)
end

# Stateless iterator over all states reachable from (0,0) using only ↑, ↓ until terminal
struct ExcursionStateSpace
    problem::ExcursionProblem
end

"""Create a stateless iterator over the state space of an `ExcursionProblem`."""
state_space(problem::ExcursionProblem) = ExcursionStateSpace(problem)

function Base.iterate(iter::ExcursionStateSpace)
    problem = iter.problem
    if problem.trajectory_length == 0
        return nothing
    end
    s = (0, 0)
    return s, s
end

function Base.iterate(iter::ExcursionStateSpace, state)
    problem = iter.problem
    x, t = state
    T = problem.trajectory_length

    if T == 0
        return nothing
    end

    if t == 0
        if T > 1
            next_state = (-1, 1)
            return next_state, next_state
        else
            return nothing
        end
    end

    if x < t
        next_state = (x + 2, t)
        return next_state, next_state
    else
        if t >= T - 1
            return nothing
        else
            t_next = t + 1
            next_state = (-t_next, t_next)
            return next_state, next_state
        end
    end
end

struct Trajectory
    states::Vector{Tuple{Int64,Int64}}
    rewards::Vector{Float64}
    actions::Vector{Int64}
end

function Trajectory(s::Tuple{Int64,Int64})
    Trajectory([s], Float64[], Int64[])
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

function append_transition(traj::Trajectory, next_state::Tuple{Int64,Int64}, action::Int64, reward::Float64)
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

function sigmoid(x::Float64)
    return 1/(1+exp(-x))
end

struct PolicyGradient
    discount:: Float64
    _policy_parameters::Dict{Tuple{Int64,Int64},Float64}
    _parameter_gradients::Dict{Tuple{Int64,Int64},Float64}
end

function init_pga(pga::PolicyGradient, problem::ExcursionProblem)
    for s in state_space(problem)
        if is_terminal(problem, s)
            continue
        end
        pga._policy_parameters[s] = randn()*0.01
        pga._parameter_gradients[s] = 0.0
    end
end

function greedy_policy(pga::PolicyGradient)
    policy = Dict{Tuple{Int64,Int64},Int64}()
    for (state,value) in pga._policy_parameters 
        if sigmoid(value) > 0.5
            policy[state] = 2
        else 
            policy[state] = 1
        end
    end
    return policy
end

function sample_action(pga::PolicyGradient, state)
    p_up = sigmoid(pga._policy_parameters[state])
    action = rand() < p_up ? 2 : 1
    return action    
end

function _reset_gradients(pga::PolicyGradient)
    for state in keys(pga._parameter_gradients)
        pga._parameter_gradients[state] = 0.0
    end
end

function sample_trajectory!(pga::PolicyGradient, traj::Trajectory, problem::ExcursionProblem, s0::Tuple{Int64,Int64})
    empty!(traj.states)
    empty!(traj.actions)
    empty!(traj.rewards)
    push!(traj.states, s0)

    while true
        current_state = traj.states[end]
        if is_terminal(problem, current_state)
            break
        end
        action = sample_action(pga, current_state)
        ns = next_state(problem, current_state, action)
        p_up = sigmoid(pga._policy_parameters[current_state])
        p_action = action == 2 ? p_up : 1-p_up
        # 0.5 is the uniform probability
        r = reward(problem, current_state, action, ns) - log(p_action/0.5)
        append_transition(traj, ns, action, r)
    end
end

function accumulate_gradient(pga::PolicyGradient, state, action, return_to_go)
    probs = sigmoid(pga._policy_parameters[state])
    a_binary = action - 1 #sigmoid expects range 0-1 (binary function)
    pga._parameter_gradients[state] += (a_binary - probs) * return_to_go
end

function train!(pga::PolicyGradient, problem::ExcursionProblem, epochs::Int64, learning_rate::Float64, batch_size::Int64)
    avg_returns = Float64[]
    s0 = starting_state(problem)
    traj = Trajectory(s0)  

    for i in 1:epochs
        if i % 10 == 0
            percent_done = i*100/epochs 
            current_reward = isempty(avg_returns)  ? "no reward yet" : avg_returns[end] 
            println("completed $i / $epochs samples. $percent_done% complete.")
            println("current reward: $current_reward")
        end
        _reset_gradients(pga)
        total_return = 0.0
        for _ in 1:batch_size 
            sample_trajectory!(pga, traj, problem, s0)
            pass = transitions(traj, pga.discount)
            final_return =  pass[1][5]
            for row in pass
                accumulate_gradient(pga, row[1], row[2], row[5])
            end
            total_return += final_return
        end
        average_return = total_return / batch_size
        push!(avg_returns, average_return)

        for state in keys(pga._parameter_gradients)
            pga._policy_parameters[state] += (learning_rate / batch_size) * pga._parameter_gradients[state]
        end
    end
    return avg_returns  
end

function greedy_trajectory_xs(greedy::Dict, problem::ExcursionProblem)
    s = starting_state(problem)
    xs = [s[1]]
    while !is_terminal(problem, s)
        a = get(greedy, s, 1)
        s = next_state(problem, s, a)
        push!(xs, s[1])
    end
    return xs
end

function sampled_trajectory_xs(pga::PolicyGradient, problem::ExcursionProblem)
    s = starting_state(problem)
    xs = [s[1]]
    while !is_terminal(problem, s)
        a = sample_action(pga, s)
        s = next_state(problem, s, a)
        push!(xs, s[1])
    end
    return xs
end

function main()
    Random.seed!(67)
    T = 300
    bias = 50.0
    R = Random.randn(Float64, 2T+1, T)
    #R = zeros( 2T+1, T)
    R[1:T+1, :] .-= 10
    R[:, T] .= (-T:T) .^ 2 .* (-bias)

    γ = 0.9
    α = 0.1

    problem = ExcursionProblem(R, T)
    params = Dict{Tuple{Int64,Int64},Float64}()  
    gradients = Dict{Tuple{Int64,Int64},Float64}()
    pga = PolicyGradient(γ, params, gradients)
    init_pga(pga, problem)
    epochs = 500
    batch_size = 32
    avg_returns = train!(pga, problem, epochs, α, batch_size)

    greedy = greedy_policy(pga)

    n_samples = 30
    ts = 0:T
    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="t", ylabel="x", title="Rare Event Trajectories")

        for i in 1:n_samples
            xs = sampled_trajectory_xs(pga, problem)
            lines!(ax, collect(ts), xs, color=(:blue, 0.20), linewidth=1,
                label= i == 1 ? "Sampled (n=$n_samples)" : nothing)
        end

        # greedy_xs = greedy_trajectory_xs(greedy, problem)
        # lines!(ax, collect(ts), greedy_xs, color=:red, linewidth=2.5, label="Greedy")

        axislegend(ax, unique=true)
        save("Rare_events.pdf",fig)
        fig
    end
    display(fig)

    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="Epochs", ylabel="Rewards", title="Rewards")
        x_ax = 1:epochs

        lines!(ax, collect(x_ax), avg_returns, color=:red, linewidth=2.5, label="avg returns")

        axislegend(ax, unique=true)  
        save("Rewards.pdf",fig)
        fig
    end
    display(fig)


    #return pga, problem, avg_returns, greedy
    return avg_returns
end