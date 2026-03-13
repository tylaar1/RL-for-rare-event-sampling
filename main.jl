using CairoMakie
using ProgressBars
using Random
#include("plotters.jl") #TODO fix load order to avoid undef var error 

# Training struct
struct ExcursionProblem
    rewards::Matrix{Float64}
    trajectory_length::Int
    γ::Float64
end

function reward(problem::ExcursionProblem, s′)
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
function starting_state(problem::ExcursionProblem)
    return (0, 0)
end

# Stateless iterator over all states reachable from (0,0) using only ↑, ↓ until terminal
struct ExcursionStateSpace
    problem::ExcursionProblem
end

Base.eltype(::Type{ExcursionStateSpace}) = Tuple{Int,Int}

Base.IteratorEltype(::Type{ExcursionStateSpace}) = Base.HasEltype()

Base.IteratorSize(::Type{ExcursionStateSpace}) = Base.HasLength()

function Base.length(iter::ExcursionStateSpace)
    T = iter.problem.trajectory_length
    return T*(T+1) ÷ 2
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
        r = reward(problem, ns) - log(p_action/0.5)
        append_transition(traj, ns, action, r)
    end
end

function accumulate_gradient(pga::PolicyGradient, state, action, return_to_go)
    probs = sigmoid(pga._policy_parameters[state])
    a_binary = action - 1 #sigmoid expects range 0-1 (binary function)
    pga._parameter_gradients[state] += (a_binary - probs) * return_to_go
end

struct ExactSolution
    values::Dict{Tuple{Int64,Int64},Float64}
    policy::Dict{Tuple{Int64,Int64},Float64}
end

function train!(pga::PolicyGradient, problem::ExcursionProblem,solution::ExactSolution, epochs::Int64, learning_rate::Float64,batch_size::Int64,LOG_INTERVAL::Int64)
    avg_returns = Float64[]
    s0 = starting_state(problem)
    traj = Trajectory(s0)  
    D_kl = Float64[]
    for i in 1:epochs
        if i % LOG_INTERVAL == 0
            percent_done = i*100/epochs 
            current_reward = isempty(avg_returns)  ? 0.0 : avg_returns[end] 
            println("completed $i / $epochs samples. $percent_done% complete.")
            println("most recent returns: $current_reward")
            KL_divergence = kl_divergence(problem,pga,solution)
            push!(D_kl,KL_divergence)
            println("Current KL Divergence: $KL_divergence")
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
    return avg_returns, D_kl  
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



function calculate_policy!(problem::ExcursionProblem,solution::ExactSolution,s)
    x,t = s
    s_prime_up = x+1,t+1
    s_prime_down = x-1,t+1
    if t == problem.trajectory_length - 1
        theta = reward(problem,s_prime_up)-reward(problem,s_prime_down) #no value function for final state, ln terms cancel
    else    
        theta = (reward(problem,s_prime_up)+(problem.γ*solution.values[s_prime_up]))-(reward(problem,s_prime_down)+(problem.γ*solution.values[s_prime_down])) #no value function for final state, ln terms cancel
    end
    p_up   = sigmoid(theta)
    p_down = 1.0 - p_up

    up_entropy   =  log(p_up   / 0.5) 
    down_entropy =  log(p_down / 0.5) 
    if t == problem.trajectory_length - 1
        V = (p_up*(reward(problem,s_prime_up)-up_entropy)) + ((p_down)*(reward(problem,s_prime_down)-down_entropy))
    else    
        V = (p_up*(reward(problem,s_prime_up)-up_entropy+(problem.γ*solution.values[s_prime_up]))) + ((p_down)*(reward(problem,s_prime_down)-down_entropy+(problem.γ*solution.values[s_prime_down])))
    end
    solution.values[s] = V
    solution.policy[s] = theta
end



function kl_divergence(problem::ExcursionProblem,pga::PolicyGradient,solution::ExactSolution)
    #TODO add guard against NaNs
    D_kl = 0.0
    T = problem.trajectory_length
    total_p_theta = 0.0
    total_p_w = 0.0
    for i in 0:2^T-1
        actions = i
        s = starting_state(problem)
        p_theta_w = 1.0
        p_exact_w = 1.0 
        for _ in 1:T 
            a = actions%2
            a += 1 #action space is 1/2 not 0/1
            actions = actions >> 1
            s_prime = next_state(problem,s,a)
            p_up_theta = sigmoid(pga._policy_parameters[s])
            p_up_exact = sigmoid(solution.policy[s]) 

            p_theta_w *= a == 2 ? p_up_theta : 1.0 - p_up_theta
            p_exact_w *= a == 2 ? p_up_exact : 1.0 - p_up_exact
            s = s_prime
        end
        total_p_theta += p_theta_w
        total_p_w += p_exact_w
        D_kl += p_theta_w * log(p_theta_w/p_exact_w)
    end
    return D_kl
end


function main()
    Random.seed!(67)
    T = 12
    bias = 0.5
    R = Random.randn(Float64, 2T+1, T)
    #R = zeros( 2T+1, T)
    R[1:T+1, :] .-= 1
    R[:, T] .= (-T:T) .^ 2 .* (-bias)

    γ = 1.0
    α = 0.1

    values = Dict{Tuple{Int64,Int64},Float64}()  
    policy = Dict{Tuple{Int64,Int64},Float64}() 

    problem = ExcursionProblem(R, T, γ)
    params = Dict{Tuple{Int64,Int64},Float64}()  
    gradients = Dict{Tuple{Int64,Int64},Float64}()
    pga = PolicyGradient(γ, params, gradients)
    solution = ExactSolution(values,policy)
    init_pga(pga, problem)

    for s in state_space(problem)
        solution.values[s] = 0.0
        solution.policy[s] = 0.0
    end
    for s in reverse(collect(state_space(problem)))
        calculate_policy!(problem,solution,s)
    end
    epochs = 10000
    batch_size = 64
    LOG_INTERVAL = 100
    avg_returns, D_kl = train!(pga, problem,solution, epochs, α, batch_size,LOG_INTERVAL)


    #***Comment/Uncomment plotting functions based on need***
    #plot_trajectories(pga,problem)
    #plot_returns(solution,epochs,avg_returns)
    #plot_policy_comparison(pga,solution,problem)
    #plot_kl_divergence(D_kl,LOG_INTERVAL,epochs)
    
    #return pga, problem, avg_returns, greedy
    #return avg_returns
end