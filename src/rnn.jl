using LinearAlgebra, NNlib, Enzyme, Optimisers, Random, Statistics

const Reverse = Enzyme.Reverse #as multple packages export reverse

struct ExcursionProblem
    rewards::Array{Float64,3}
    trajectory_length::Int
    γ::Float64
end

struct ExcursionStateSpace
    problem::ExcursionProblem
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




# RNN
function construct_RNN(hidden_dim::Int; input_dim::Int = 1)   #returns tuple of matricies, note we assume input dim==output dim here   
    init(m, n) = (rand(m, n) .- 0.5) .* 0.2
    return (V = init(input_dim, hidden_dim),           # hidden -> logit
            W = init(hidden_dim, hidden_dim),  # recurrent
            U = init(hidden_dim, input_dim),   # input -> hidden
            B = zeros(hidden_dim))
end


function step!(h, p, hprev, x,return_h=false) #performs one set of computations from input to output
    mul!(h, p.W, hprev)
    h .+= view(p.U, :, 1) .* x .+ p.B
    h .= tanh.(h)
    if return_h == false
        return dot(view(p.V, 1, :), h)
    else
        return dot(view(p.V, 1, :), h), h
    end
end

# Replay: deterministic, no rand. hs[:, 1] = h_0 = 0, hs[:, t+1] = h_t.
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


# weights[t] = discount_t * advantage_t / batch. xs, actions, weights are all constants.
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

# ───────────── training ─────────────
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
        Gsum  = sum(Gs)                         # elementwise over time

        dp = Enzyme.make_zero(p)                # shadows ACCUMULATE, so reuse across the batch
        for (τ, G) in zip(trajs, Gs)
            baseline = (Gsum .- G) ./ (batch - 1)           # leave-one-out, per time step
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

# ───────────── checks ─────────────
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


function recursion(sequence,dict,T)
    t = length(sequence)
    s_prev = pop!(sequence.copy())
    h_prev = dict[s_prev]
    h = zeros(length(p.B))
    x = sequence[last]
    step!(h,p,h_prev,x)
    if t == T
        return dict
    else 
        up_seq = push!(sequence.copy(),1)
        down_seq = push!(sequence.copy(),-1)
        return recursion(up_seq,dict,T)
        return recursion(down_seq,dict,T)
    end
end

function get_hidden_states(p,problem)
    states_dict = Dict{Tuple,Matrix}()
    x = 0
    t = 1
    hs = zeros(length(p.B),T+1)
    
end
    
T = 4
problem = ExcursionProblem(def_problem(T, 0.5, -1.0), T, 1.0)
p0 = construct_RNN(16)
#check_replay(problem, p0)
#fd_check(problem, p0)
p = trainRNN(problem, 200, 10)
get_hidden_states(p,problem)