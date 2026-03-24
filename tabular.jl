"""setup iterator"""
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

"""Iterator done"""

function tab_sigmoid(x::Float64) #rename to avoid conflict with sig function in Lux
    return 1/(1+exp(-x))
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
        if tab_sigmoid(value) > 0.5
            policy[state] = 2
        else 
            policy[state] = 1
        end
    end
    return policy
end

function sample_action(pga::PolicyGradient, state) #tabular version
    p_up = tab_sigmoid(pga._policy_parameters[state])
    action = rand() < p_up ? 2 : 1
    return action    
end

function _reset_gradients(pga::PolicyGradient)
    for state in keys(pga._parameter_gradients)
        pga._parameter_gradients[state] = 0.0
    end
end

function sample_trajectory!(pga::PolicyGradient, traj::Trajectory, problem::ExcursionProblem, s0::Tuple{Int64,Int64}) #tabular version
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
        p_up = tab_sigmoid(pga._policy_parameters[current_state])
        p_action = action == 2 ? p_up : 1-p_up
        # 0.5 is the uniform probability
        r = reward(problem, ns) - log(p_action/0.5)
        append_transition(traj, ns, action, r)
    end
end

function accumulate_gradient(pga::PolicyGradient, state, action, return_to_go)
    probs = tab_sigmoid(pga._policy_parameters[state])
    a_binary = action - 1 #sigmoid expects range 0-1 (binary function)
    pga._parameter_gradients[state] += (a_binary - probs) * return_to_go
end

function train!(pga::PolicyGradient, problem::ExcursionProblem,solution::ExactSolution, epochs::Int64, learning_rate::Float64,batch_size::Int64,LOG_INTERVAL::Int64)
    avg_returns = Float64[]
    s0 = starting_state(problem)
    traj = Trajectory()  
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

function greedy_trajectory_xs(greedy::Dict, problem::ExcursionProblem) #unused?
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


"""Exact Solution"""
function calculate_policy!(problem::ExcursionProblem,solution::ExactSolution,s)
    x,t = s
    s_prime_up = x+1,t+1
    s_prime_down = x-1,t+1
    if t == problem.trajectory_length - 1
        theta = reward(problem,s_prime_up)-reward(problem,s_prime_down) #no value function for final state, ln terms cancel
    else    
        theta = (reward(problem,s_prime_up)+(problem.γ*solution.values[s_prime_up]))-(reward(problem,s_prime_down)+(problem.γ*solution.values[s_prime_down])) #no value function for final state, ln terms cancel
    end
    p_up   = tab_sigmoid(theta)
    p_down = 1.0 - p_up

    up_entropy   =  log(p_up   / 0.5) 
    down_entropy =  log(p_down / 0.5) 
    if t == problem.trajectory_length - 1
        V = ((p_up)*(reward(problem,s_prime_up)-up_entropy)) + ((p_down)*(reward(problem,s_prime_down)-down_entropy))
    else    
        V = ((p_up)*(reward(problem,s_prime_up)-up_entropy+(problem.γ*solution.values[s_prime_up]))) + ((p_down)*(reward(problem,s_prime_down)-down_entropy+(problem.γ*solution.values[s_prime_down])))
    end
    solution.values[s] = V
    solution.policy[s] = theta
end



function kl_divergence(problem::ExcursionProblem,pga::PolicyGradient,solution::ExactSolution) #can adapt to also apply to neural net?
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
            p_up_theta = tab_sigmoid(pga._policy_parameters[s])
            p_up_exact = tab_sigmoid(solution.policy[s]) 

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
