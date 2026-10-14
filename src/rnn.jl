using LinearAlgebra, NNlib, Enzyme, Optimisers, Random, Statistics, JLD2, LogExpFunctions

const Reverse = Enzyme.Reverse #as multple packages export reverse

struct ExcursionProblem
    rewards::Array{Float64,3}
    trajectory_length::Int
    γ::Float64
end

struct ExcursionStateSpace
    problem::ExcursionProblem
end

struct ExactSolution
    values::Dict{Tuple{Int64,Int64,Int64},Float64}
    policy::Dict{Tuple{Int64,Int64,Int64},Float64}
end

function reward(problem::ExcursionProblem, s′)
    x′, t′, T = s′
    problem.rewards[x′ + T + 1, t′, T]
end

function def_problem(T::Int64, bias::Float64, negative_penalty::Float64)
    R = zeros(2T + 1, T, T)
    R[1:T, :, T] .+= negative_penalty         # intermediate reward
    R[:, T, T] .= (-T:T) .^ 2 .* (-bias)      # terminal reward
    return R
end

function starting_state(problem::ExcursionProblem)
    T = problem.trajectory_length
    return (0, 0,T)
end

function next_state(::ExcursionProblem, s, a)
    x, t ,T = s
    (Int64(x + 2a - 3), t+1,T) 
end


# RNN
function construct_RNN(hidden_dim::Int; input_dim::Int = 1)   #returns tuple of matricies, note we assume input dim==output dim here   
    init(m, n) = (rand(m, n) .- 0.5) .* 0.2
    return (V = init(input_dim, hidden_dim),           # hidden -> logit
            W = init(hidden_dim, hidden_dim),  # recurrent
            U = init(hidden_dim, input_dim),   # input -> hidden
            B = zeros(hidden_dim))
end


function step!(h, p, hprev, x) #modifies h in place, returns y
    mul!(h, p.W, hprev)
    h .+= view(p.U, :, 1) .* x .+ p.B
    h .= tanh.(h)
    return dot(view(p.V, 1, :), h)
end

function forward!(zs, hs, p, xs)
    for t in eachindex(zs)
        zs[t] = step!(view(hs, :, t + 1), p, view(hs, :, t), xs[t])
    end
    return nothing
end


function rollout(problem::ExcursionProblem, p)
    T = problem.trajectory_length
    #data storage
    n = length(p.B)  #length of hidden dim
    hs = zeros(n, T + 1) 
    zs = zeros(T); xs = zeros(T); actions = zeros(T)
    rewards = zeros(T); logps = zeros(T)
    x = 0.0
    state = 0
    for t in 1:T
        xs[t] = x
        zs[t] = step!(view(hs, :, t + 1), p, view(hs, :, t), x)
        a = rand() < sigmoid(zs[t]) ? 1.0 : 0.0
        logpa = a == 1.0 ? logsigmoid(zs[t]) : logsigmoid(-zs[t])
        state += a == 1.0 ? 1 : -1
        rewards[t] = reward(problem, (state, t, T)) - (logpa - log(0.5))
        actions[t] = a
        logps[t] = logpa
        x = 2a - 1                              
    end
    return (; xs, actions, rewards, logps, zs)
end

function discounted_returns(rewards, γ)
    G = similar(rewards)
    acc = 0.0
    for t in length(rewards):-1:1
        acc = rewards[t] + γ * acc
        G[t] = acc
    end
    return G
end


function surrogate!(zs, hs, p, xs, actions, weights)
    forward!(zs, hs, p, xs)
    L = 0.0
    for t in eachindex(zs)
        a = actions[t]
        logp = a * logsigmoid(zs[t]) + (1 - a) * logsigmoid(-zs[t])
        L -= logp * weights[t]
    end
    return L
end


function trainRNN(problem::ExcursionProblem, epochs::Int, LOG_INTERVAL::Int;
                  hidden = 16, batch = 32, lr = 0.01, clip = 1.0)
    @assert batch ≥ 2                           # baseline needs another traj to compare against
    T = problem.trajectory_length
    γ = problem.γ
    p = construct_RNN(hidden)
    opt = Optimisers.setup(OptimiserChain(ClipNorm(clip), Adam(lr)), p)
    discount = γ .^ (0:T-1)                     

    for epoch in 1:epochs
        trajs = [rollout(problem, p) for _ in 1:batch]
        Gs    = [discounted_returns(τ.rewards, γ) for τ in trajs]
        Gsum  = sum(Gs)                         

        dp = Enzyme.make_zero(p)                # reset every batvh
        for (τ, G) in zip(trajs, Gs)
            baseline = (Gsum .- G) ./ (batch - 1)           # -1 as we subtract G
            weights  = discount .* (G .- baseline) ./ batch
            zs = zeros(T); hs = zeros(hidden, T + 1)
            dzs = zero(zs); dhs = zero(hs)
            Enzyme.autodiff(Reverse, surrogate!, Active,
                Duplicated(zs, dzs), Duplicated(hs, dhs), Duplicated(p, dp),
                Const(τ.xs), Const(τ.actions), Const(weights))
        end

        gnorm = sqrt(sum(sum(abs2, g) for g in values(dp)))
        opt, p = Optimisers.update(opt, p, dp)

        if epoch % LOG_INTERVAL == 0
            mean_total = mean(sum(τ.rewards) for τ in trajs)
            @info "epoch $epoch" mean_total gnorm
        end
    end
    return p
end

# ───────────── checks (AI genrated) ─────────────
function check_replay(problem, p)               # replay must reproduce the rollout's logits
    τ = rollout(problem, p)
    T = problem.trajectory_length
    zs = zeros(T); hs = zeros(length(p.B), T + 1)
    forward!(zs, hs, p, τ.xs)
    @assert zs ≈ τ.zs
end

function fd_check(problem, p; ε = 1e-6)         # Enzyme vs central finite differences
    τ = rollout(problem, p)
    T = problem.trajectory_length; n = length(p.B)
    w = randn(T)
    f(q) = surrogate!(zeros(T), zeros(n, T + 1), q, τ.xs, τ.actions, w)
    dp = Enzyme.make_zero(p)
    Enzyme.autodiff(Reverse, surrogate!, Active,
        Duplicated(zeros(T), zeros(T)), Duplicated(zeros(n, T + 1), zeros(n, T + 1)),
        Duplicated(p, dp), Const(τ.xs), Const(τ.actions), Const(w))
    q = deepcopy(p)
    for k in keys(p)
        A = getfield(q, k); gfd = similar(A)
        for i in eachindex(A)
            old = A[i]
            A[i] = old + ε; fp = f(q)
            A[i] = old - ε; fm = f(q)
            A[i] = old
            gfd[i] = (fp - fm) / (2ε)
        end
        println(k, ": rel err = ", norm(gfd - getfield(dp, k)) / norm(gfd))
    end
end



#KL divergence section

function expand!(dict_hs, dict_ys,p, sequence, T)
    length(sequence) == T + 1 && return dict_hs, dict_ys    
    h_prev = dict_hs[sequence]
    for x in (1, -1)
        push!(sequence, x)
        h = zeros(length(p.B))
        y = step!(h, p, h_prev, x) #logit not probability
        dict_hs[copy(sequence)] = h
        dict_ys[copy(sequence)] = y
        expand!(dict_hs,dict_ys, p, sequence, T)
        pop!(sequence)                          # backtrack
    end
    return dict_hs,dict_ys
end

function build_states(p, h0, T)
    seq = [0]
    y0 = dot(view(p.V, 1, :), h0)
    dict_hs = Dict{Vector{Int},typeof(h0)}(copy(seq) => h0)
    dict_ys = Dict{Vector{Int},typeof(y0)}(copy(seq) => y0)
    return expand!(dict_hs,dict_ys, p, seq, T)
end

function RNN_kl_divergence(problem::ExcursionProblem,solution::ExactSolution, ys_dict)  #simple adaption from NN kl div
    D_kl = 0.0
    T = problem.trajectory_length
    total = 0.0
    log_p_theta = 0.0
    for i in 0:2^T-1
        total += exp(log_p_theta)
        actions = i
        s = starting_state(problem)
        log_p_theta = 0.0
        log_p_exact = 0.0
        actions_list=Int[0]
        for _ in 1:T
            #theta_s = ys_dict[actions_list]
            a = actions % 2 + 1  # action space is 1/2 not 0/1
            push!(actions_list, 2*(actions % 2) - 1)
            actions >>= 1
            s_prime = next_state(problem, s, a)

            theta_s = ys_dict[actions_list]
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
    total += exp(log_p_theta)
    Scaled_D_KL = D_kl/T
    return Scaled_D_KL,total
end
    
T = 10
problem = ExcursionProblem(def_problem(T, 0.5, -1.0), T, 1.0)
hidden_dim = 16
p0 = construct_RNN(hidden_dim)
#check_replay(problem, p0)
#fd_check(problem, p0)
p = trainRNN(problem, 200, 10)
hs, ys = build_states(p,zeros(hidden_dim),T)
@load "data/solutions10-200.jld2" solutions
solution = solutions[T]
Kl = RNN_kl_divergence(problem,solution,ys)#
println(Kl)