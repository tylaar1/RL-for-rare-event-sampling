"""setup iterator"""
Base.eltype(::Type{ExcursionStateSpace}) = Tuple{Int,Int,Int}

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
    T = problem.trajectory_length
    if problem.trajectory_length == 0
        return nothing
    end
    s = (0, 0,T)
    return s, s
end

function Base.iterate(iter::ExcursionStateSpace, state)
    problem = iter.problem
    x, t, T = state
    T = problem.trajectory_length

    if T == 0
        return nothing
    end

    if t == 0
        if T > 1
            next_state = (-1, 1, T)
            return next_state, next_state
        else
            return nothing
        end
    end

    if x < t
        next_state = (x + 2, t, T)
        return next_state, next_state
    else
        if t >= T - 1
            return nothing
        else
            t_next = t + 1
            next_state = (-t_next, t_next, T)
            return next_state, next_state
        end
    end
end

"""Iterator done"""


Base.eltype(::Type{ExcursionStateSpace3D}) = Tuple{Int,Int,Int}

Base.IteratorEltype(::Type{ExcursionStateSpace3D}) = Base.HasEltype()

Base.IteratorSize(::Type{ExcursionStateSpace3D}) = Base.HasLength()

function Base.length(iter::ExcursionStateSpace3D)
    T = iter.problem.trajectory_length
    return T*(T+1) ÷ 2
end

"""Not actually 3D just an adaption that takes `ExcursionProblem3D`, `T` as inputs."""
state_space(problem::ExcursionProblem3D,T) = ExcursionStateSpace3D(problem,T)

function Base.iterate(iter::ExcursionStateSpace3D)
    T = iter.trajectory_length
    if T == 0
        return nothing
    end
    s = (0, 0, T)
    return s, s
end

function Base.iterate(iter::ExcursionStateSpace3D, state)
    x, t, T = state
    T = iter.trajectory_length

    if T == 0
        return nothing
    end

    if t == 0
        if T > 1
            next_state = (-1, 1, T)
            return next_state, next_state
        else
            return nothing
        end
    end

    if x < t
        next_state = (x + 2, t, T)
        return next_state, next_state
    else
        if t >= T - 1
            return nothing
        else
            t_next = t + 1
            next_state = (-t_next, t_next, T)
            return next_state, next_state
        end
    end
end

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
    policy = Dict{Tuple{Int64,Int64,Int64},Int64}()
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

function sample_trajectory!(pga::PolicyGradient, traj::Trajectory, problem::ExcursionProblem, s0::Tuple{Int64,Int64,Int64}) #tabular version
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

function train!(pga::PolicyGradient, problem::ExcursionProblem, epochs::Int64, learning_rate::Float64,batch_size::Int64,LOG_INTERVAL::Int64,solutions)
    avg_returns = Float64[]
    s0 = starting_state(problem)
    traj = Trajectory()  
    D_kl = Float64[]
    T = problem.trajectory_length
    solution = solutions[T]
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


function calculate_policy!(problem::ExcursionProblem, solution::ExactSolution, s)
    x, t, T = s
    s_prime_up   = (x+1, t+1, T)
    s_prime_down = (x-1, t+1, T)

    final_step = t == T - 1
                                                        #optional term included for non final steps
    Q_up   = reward(problem, s_prime_up)   + (final_step ? 0.0 : problem.γ * solution.values[s_prime_up])
    Q_down = reward(problem, s_prime_down) + (final_step ? 0.0 : problem.γ * solution.values[s_prime_down])

    theta = Q_up - Q_down  

    p_up   = logistic(theta)
    p_down = 1.0 - p_up  # or logistic(-theta) if theta is very large

    entropy_up   = -log1pexp(-theta) + log(2)
    entropy_down = -log1pexp( theta) + log(2)

    V = p_up   * (Q_up   - entropy_up)   +
        p_down * (Q_down - entropy_down)

    solution.values[s] = V
    solution.policy[s] = theta
end

function calculate_local_policy!(problem::ExcursionProblem, solution::ExactSolution, s)
    x, t, T = s
    s_prime_up   = (x+1, t+1, T)
    s_prime_down = (x-1, t+1, T)

    final_step = t == T - 1

    Q_up   = reward(problem, s_prime_up)   
    Q_down = reward(problem, s_prime_down) 

    theta = Q_up - Q_down  

    p_up   = logistic(theta)
    p_down = 1.0 - p_up  # or logistic(-theta) if theta is very large

    entropy_up   = -log1pexp(-theta) + log(2)
    entropy_down = -log1pexp( theta) + log(2)

    V = p_up   * (Q_up   - entropy_up)   +
        p_down * (Q_down - entropy_down)

    solution.values[s] = V
    solution.policy[s] = theta
end



function kl_divergence(problem::ExcursionProblem,pga::PolicyGradient,solution::ExactSolution)
    D_kl = 0.0
    T = problem.trajectory_length
    for i in 0:2^T-1
        actions = i
        s = starting_state(problem)
        log_p_theta = 0.0
        log_p_exact = 0.0
        for _ in 1:T
            a = actions % 2 + 1  # action space is 1/2 not 0/1
            actions >>= 1
            s_prime = next_state(problem, s, a)

            theta_s = pga._policy_parameters[s]
            exact_s = solution.policy[s]

            if a == 2 #swapped for equivelant functions in log terms for numerical stability
                log_p_theta += -log1pexp(-theta_s)   # log(sigmoid(θ))
                log_p_exact  += -log1pexp(-exact_s)
            else
                log_p_theta += -log1pexp( theta_s)   # log(1 - sigmoid(θ)) = log(sigmoid(-θ))
                log_p_exact  += -log1pexp( exact_s)
            end
            s = s_prime
        end
        # guard: if log_p_theta = -Inf, contribution is 0
        if isfinite(log_p_theta)
            D_kl += exp(log_p_theta) * (log_p_theta - log_p_exact)
        end
    end
    return D_kl
end

function efficient_kl(problem::ExcursionProblem, pga::PolicyGradient, solution::ExactSolution)
    T = problem.trajectory_length
    f = Dict{Any, Float64}()

    for t in T:-1:0
        for x in -t:2:t
            s = (x, t, T)
            if t == T
                f[s] = 0.0
                continue
            end
            f_s = 0.0
            for a in 1:2
                s_next = next_state(problem, s, a)

                θ_s     = pga._policy_parameters[s]
                exact_s = solution.policy[s]

                if a == 2
                    log_π_a = -log1pexp(-θ_s)   # log(σ(θ))
                    log_π_b = -log1pexp(-exact_s)
                else
                    log_π_a = -log1pexp(θ_s)    # log(1 - σ(θ))
                    log_π_b = -log1pexp(exact_s)
                end

                π_a = exp(log_π_a)
                f_s += π_a * ((log_π_a - log_π_b) + f[s_next])
            end
            f[s] = f_s
        end
    end

    s0 = starting_state(problem)
    return f[s0]/T
end

function efficient_kl(problem::ExcursionProblem, approx_sol::ExactSolution, solution::ExactSolution)
    T = problem.trajectory_length
    f = Dict{Any, Float64}()

    for t in T:-1:0
        for x in -t:2:t
            s = (x, t, T)
            if t == T
                f[s] = 0.0
                continue
            end
            f_s = 0.0
            for a in 1:2
                s_next = next_state(problem, s, a)

                θ_s     = approx_sol.policy[s]
                exact_s = solution.policy[s]

                if a == 2
                    log_π_a = -log1pexp(-θ_s)   # log(σ(θ))
                    log_π_b = -log1pexp(-exact_s)
                else
                    log_π_a = -log1pexp(θ_s)    # log(1 - σ(θ))
                    log_π_b = -log1pexp(exact_s)
                end

                π_a = exp(log_π_a)
                f_s += π_a * ((log_π_a - log_π_b) + f[s_next])
            end
            f[s] = f_s
        end
    end

    s0 = starting_state(problem)
    return f[s0]/T
end

function unbiased_kl()
    T_min = 10
    T_max = 200
    T_array = [T for T in T_min:T_max if T % 2 == 0]
    kls = Dict{Int64,Float64}()
    bias = 5.0 #should be a positive val
    negative_penalty = -10.0 #should be a negative val
    γ = 1.0
    for T in T_array
        R = def_problem(T,bias,negative_penalty)
        problem = ExcursionProblem(R, T, γ)
        params = Dict{Tuple{Int64,Int64,Int64},Float64}()  
        gradients = Dict{Tuple{Int64,Int64,Int64},Float64}()
        pga = PolicyGradient(γ, params, gradients)
        for s in state_space(problem)
            pga._policy_parameters[s]=0.5
        end
        unbiased_kl = kl_divergence(problem,pga,solutions[T])
        println(unbiased_kl)
        kls[T] = unbiased_kl
    end
    @save "data/unbiased10-200.jld2" kls
    return kls
end