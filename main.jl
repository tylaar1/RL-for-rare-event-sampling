using CairoMakie
using ProgressBars
using Random

# Training struct
struct ExcursionProblem
    rewards::Matrix{Float64}
    trajectory_length::Int
    γ::Float64
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

    # We only iterate over non-terminal states t in 0:(T-1)
    if T == 0
        return nothing
    end

    if t == 0
        # After the initial state (0,0), move to t=1, x=-1 if horizon allows
        if T > 1
            next_state = (-1, 1)
            return next_state, next_state
        else
            return nothing
        end
    end

    # For t ≥ 1, reachable x are -t, -t+2, ..., t
    if x < t
        next_state = (x + 2, t)
        return next_state, next_state
    else
        # x == t, advance to next time layer if it exists
        if t >= T - 1
            return nothing
        else
            t_next = t + 1
            next_state = (-t_next, t_next)
            return next_state, next_state
        end
    end
end


# Algorithm
struct WatkinsQLearning{AQ}
    Q::AQ
    alpha::Float64
    epsilon::Float64
end
function get_q_value(alg::WatkinsQLearning, problem::ExcursionProblem, s, a)
    x, t = s 
    Q = alg.Q # A matrix
    return Q[x + problem.trajectory_length + 1, t+1, a]
end
function set_q_value!(alg::WatkinsQLearning, problem::ExcursionProblem, s, a, new_value)
    x, t = s 
    Q = alg.Q # A matrix
    Q[x + problem.trajectory_length + 1, t+1, a] = new_value
    new_value
end
function greedy_action(alg::WatkinsQLearning, problem::ExcursionProblem, s)
    x, t = s 
    Q = alg.Q # A matrix
    state_q_values = @views Q[x + problem.trajectory_length + 1, t+1, :]
    best_action = argmax(state_q_values)
    return best_action
end
function value(alg::WatkinsQLearning, problem::ExcursionProblem, s)
    x, t = s 
    Q = alg.Q # A matrix
    state_q_values = @views Q[x + problem.trajectory_length + 1, t+1, :]
    best_value = maximum(state_q_values)
    return best_value
end
function epsilon_greedy_action(alg::WatkinsQLearning, problem, s)
    if rand() < alg.epsilon
        return Random.rand(action_space(problem, s))
    end
    return greedy_action(alg, problem, s)
end
function step_q_function!(alg::WatkinsQLearning, problem::ExcursionProblem, state, action, next_state, reward)
    α = alg.alpha
    γ = problem.γ

    current_q_value = get_q_value(alg, problem, state, action)
    target = reward + γ * value(alg, problem, next_state)

    set_q_value!(alg, problem, state, action, (1-α) * current_q_value + α * target)
    nothing
end

# Exact dynamic programming solution (deterministic environment)
struct ExactExcursionAlgorithm{AV, AP}
    values::AV
    policy::AP
end


function value(alg::ExactExcursionAlgorithm, problem::ExcursionProblem, s)
    x, t = s
    return alg.values[x + problem.trajectory_length + 1, t + 1]
end
function set_value!(alg::ExactExcursionAlgorithm, problem::ExcursionProblem, s, new_value)
    x, t = s
    alg.values[x + problem.trajectory_length + 1, t + 1] = new_value
end
function get_policy(alg::ExactExcursionAlgorithm, problem::ExcursionProblem, s)
    x, t = s
    return alg.policy[x + problem.trajectory_length + 1, t + 1]
end
function set_policy!(alg::ExactExcursionAlgorithm, problem::ExcursionProblem, s, new_policy)
    x, t = s
    alg.policy[x + problem.trajectory_length + 1, t + 1] = new_policy
end

function greedy_action(alg::ExactExcursionAlgorithm, problem::ExcursionProblem, s)
    return argmax(a -> begin
        s′ = next_state(problem, s, a)
        r = reward(problem, s, a, s′)
        next_value = value(alg, problem, s′)
        return r + problem.γ * next_value
    end, action_space(problem, s))
end

function iterate_values!(alg::ExactExcursionAlgorithm, problem::ExcursionProblem)
    has_changed = false
    for s in state_space(problem)
        if is_terminal(problem, s)
            continue # skip terminal states
        end

        a = get_policy(alg, problem, s)
        s′ = next_state(problem, s, a)
        r = reward(problem, s, a, s′)
        next_value = r + problem.γ * value(alg, problem, s′)

        current_value = value(alg, problem, s)
        if next_value != current_value
            set_value!(alg, problem, s, next_value)
            has_changed = true
        end
    end
    return has_changed
end
function iterate_policy!(alg::ExactExcursionAlgorithm, problem::ExcursionProblem)
    has_changed = false
    for s in state_space(problem)
        if is_terminal(problem, s)
            continue # skip terminal states
        end

        existing_a = get_policy(alg, problem, s)
        new_a = greedy_action(alg, problem, s)

        if new_a != existing_a
            has_changed = true
            set_policy!(alg, problem, s, new_a)
        end
    end
    return has_changed
end

function solve!(alg::ExactExcursionAlgorithm, problem::ExcursionProblem)
    while true
        while iterate_values!(alg, problem) end

        if !iterate_policy!(alg, problem) # False when no policy has been updated
            break
        end
    end
    nothing
end

function exact_solution(problem::ExcursionProblem)
    values = zeros(2*problem.trajectory_length+1, problem.trajectory_length+1)
    policy = ones(Int, 2*problem.trajectory_length+1, problem.trajectory_length+1)

    alg = ExactExcursionAlgorithm(values, policy)
    solve!(alg, problem)
    return alg
end

# Training function
function train!(alg::WatkinsQLearning, problem::ExcursionProblem; steps::Int=1000)
    s = starting_state(problem)
    for _ in 1:steps
        # Sample action and next state
        a = epsilon_greedy_action(alg, problem, s)
        s′ = next_state(problem, s, a)
        r = reward(problem, s, a, s′)

        # Actually learn from the experience
        step_q_function!(alg, problem, s, a, s′, r)

        # Transition to next state
        if is_terminal(problem, s′)
            s = starting_state(problem)
        else
            s = s′
        end
    end
end

# Diagnostic function
function score_trajectory(alg, problem::ExcursionProblem)
    s = starting_state(problem)
    return score_trajectory(alg, problem, s)
end
function score_trajectory(alg, problem::ExcursionProblem, s)
    total_reward = 0
    discount = 1.0
    while !is_terminal(problem, s)
        a = greedy_action(alg, problem, s)
        s′ = next_state(problem, s, a)
        r = reward(problem, s, a, s′)
        
        total_reward += r * discount
        discount *= problem.γ
        # Update the state
        s = s′
    end
    return total_reward
end

# Sample Trajectory
function trajectory(alg, problem::ExcursionProblem)
    s = starting_state(problem)
    total_reward = 0
    discount = 1.0
    positions = zeros(Int, problem.trajectory_length+1)
    positions[1] = 0
    t = 0
    while !is_terminal(problem, s)
        a = greedy_action(alg, problem, s)
        s′ = next_state(problem, s, a)
        r = reward(problem, s, a, s′)
        
        total_reward += r * discount
        discount *= problem.γ
        # Update the state
        s = s′
        t += 1
        positions[t+1] = s[1] # s[1] is the position (x, t) is the state
    end
    return positions, total_reward
end

# Training
function main()
    Random.seed!(1234)
    # Define the problem
    T = 20
    bias = 0.5
    R = Random.randn(Float64, 2T+1, T) # (x', t')
    R[1:T, :] .-= 10 # Strongly discourage negative x values
    R[:, T] .= (-T:T) .^ 2 .* (-bias) # Set a potential well at the end time

    # Define the agent
    Q = ones(Float64, 2T + 1, T+1, 2) # (x, t, a), note that a = 2 (↑) and a = 1 (↓)

    # Define training parameters
    γ = 1.0
    α = 1.0
    ϵ = 0.2

    problem = ExcursionProblem(R, T, γ)
    alg = WatkinsQLearning(Q, α, ϵ)

    # Compute exact solution for comparison
    exact_alg = exact_solution(problem)
    best_value = value(exact_alg, problem, starting_state(problem))

    chunks = 100
    steps_per_chunk = 100

    returns = zeros(Float64, chunks+1)
    returns[begin] = score_trajectory(alg, problem)

    for i in ProgressBar(1:chunks)
        train!(alg, problem; steps=steps_per_chunk)
        # Keep track of the score of the policy over time
        returns[i+1] = score_trajectory(alg, problem)
    end

    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="Steps", ylabel="<G>", xscale=log10)
        steps_axis = 1:steps_per_chunk:(steps_per_chunk*(1+chunks))

        lines!(ax, steps_axis, returns, label="Returns")
        lines!(ax, steps_axis, fill(best_value, length(returns)); linestyle=:dash, color=:black, label="Best value")
        fig
    end
    display(fig)

    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="t", ylabel="x")

        positions, _ = trajectory(alg, problem)

        lines!(ax, 0:(length(positions)-1), positions, color=:red)


        exact_positions, _ = trajectory(exact_alg, problem)

        lines!(ax, 0:(length(exact_positions)-1), exact_positions, color=:blue, linestyle=:dash, alpha=0.8)
        fig
    end
    display(fig)

    return alg, problem, returns
end