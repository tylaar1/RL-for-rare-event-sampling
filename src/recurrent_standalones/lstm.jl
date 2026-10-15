using LinearAlgebra, NNlib, Enzyme, Optimisers, Random, Statistics, JLD2, LogExpFunctions

const Reverse = Enzyme.Reverse #as multple packages export reverse

struct ExcursionProblem
    rewards::Array{Float64,2}
    trajectory_length::Int
    γ::Float64
end

struct ExcursionStateSpace
    problem::ExcursionProblem
end

struct ExactSolution #must be 3d as that is the form that is loaded
    values::Dict{Tuple{Int64,Int64,Int64},Float64}
    policy::Dict{Tuple{Int64,Int64,Int64},Float64}
end


function reward(problem::ExcursionProblem, s′)
    T = problem.trajectory_length
    x′, t′ = s′
    problem.rewards[x′ + T + 1, t′]
end

function def_problem(T::Int64, bias::Float64, negative_penalty::Float64)
    R = zeros(2T + 1, T)
    R[1:T, :] .+= negative_penalty         # intermediate reward
    R[:, T] .= (-T:T) .^ 2 .* (-bias)      # terminal reward
    return R
end

function starting_state(problem::ExcursionProblem)
    return (0, 0)
end

function next_state(::ExcursionProblem, s, a)
    x, t = s
    (Int64(x + 2a - 3), t+1) 
end


# LSTM
function construct_LSTM(hidden_dim::Int; input_dim::Int = 1)   #returns tuple of matricies, note we assume input dim==output dim here   
    init(m, n) = (rand(m, n) .- 0.5) .* 0.2
    return (W_1 = init(4*hidden_dim,hidden_dim), #input,forget,output,candidate
            U_1 = init(4*hidden_dim,input_dim),
            B_1 = vcat(zeros(hidden_dim), ones(hidden_dim), zeros(hidden_dim), ones(hidden_dim)), #input(0),forget(1),output(0),candidate(1)
            W_2 = init(4*hidden_dim,hidden_dim), #input,forget,output,candidate
            U_2 = init(4*hidden_dim,hidden_dim), #input is output of first layer so same dim as hidden
            B_2 = vcat(zeros(hidden_dim), ones(hidden_dim), zeros(hidden_dim), ones(hidden_dim)), #input(0),forget(1),output(0),candidate(1)
            V = init(input_dim,hidden_dim), #only used for genersting logits, not for state update
            B_out = zeros(input_dim)) #only used for generating logits, not for state update
end


function step!(h_1,c_1,h_2,c_2, p, hprev_1,cprev_1, hprev_2,cprev_2, x) 
    hidden_size = length(h_1)
    #println("size of W_1: ", size(p.W_1), " size of hprev_1: ", size(hprev_1), " size of U_1: ", size(view(p.U_1, :, 1) ), " size of x: ", size(x), " size of B_1: ", size(p.B_1))
    ins_1 = p.W_1 * hprev_1 .+ view(p.U_1, :, 1) .* x .+ p.B_1
    input_gate = σ.(view(ins_1, 1:hidden_size)) 
    forget_gate = σ.(view(ins_1, hidden_size+1:2*hidden_size)) 
    output_gate = σ.(view(ins_1, 2*hidden_size+1:3*hidden_size)) 
    candidate_mem = tanh.(view(ins_1, 3*hidden_size+1:4*hidden_size)) 
    c_1 .= forget_gate .* cprev_1 .+ input_gate .* candidate_mem
    h_1 .= tanh.(c_1) .* output_gate
    #println("size of W_2: ", size(p.W_2), " size of hprev_2: ", size(hprev_2), " size of U_2: ", size(p.U_2 ), " size of h_1: ", size(h_1), " size of B_2: ", size(p.B_2))
    ins_2 = p.W_2 * hprev_2 .+ p.U_2* h_1 .+ p.B_2 #note we now have matrix multiplication for the second layer as input is now a vector of size hidden_dim, not a scalar
    input_gate = σ.(view(ins_2, 1:hidden_size)) #these variables are safe to reuse as they are not used after this point - unsure on good practice?
    forget_gate = σ.(view(ins_2, hidden_size+1:2*hidden_size))
    output_gate = σ.(view(ins_2, 2*hidden_size+1:3*hidden_size))
    candidate_mem = tanh.(view(ins_2, 3*hidden_size+1:4*hidden_size))
    c_2 .= forget_gate .* cprev_2 .+ input_gate .* candidate_mem
    h_2 .= tanh.(c_2) .* output_gate
    return dot(view(p.V, 1, :), h_2) .+ p.B_out[1] #one element vector so do scalar addition.
end

function forward!(zs, hs_1, cs_1, hs_2, cs_2, p, xs)
    for t in eachindex(zs)
        zs[t] = step!(view(hs_1, :, t + 1), view(cs_1, :, t + 1), view(hs_2, :, t + 1), view(cs_2, :, t + 1), p, view(hs_1, :, t), view(cs_1, :, t), view(hs_2, :, t), view(cs_2, :, t), xs[t])
    end
    return nothing
end


function rollout(problem::ExcursionProblem, p)
    T = problem.trajectory_length
    #data storage
    n = length(view(p.V, 1, :))  #length of hidden dim
    hs_1 = zeros(n, T + 1) 
    cs_1 = zeros(n, T + 1)
    hs_2 = zeros(n, T + 1)
    cs_2 = zeros(n, T + 1)
    zs = zeros(T); xs = zeros(T); actions = zeros(T)
    rewards = zeros(T); logps = zeros(T)
    x = 0.0
    state = 0
    for t in 1:T
        xs[t] = x
        zs[t] = step!(view(hs_1, :, t + 1), view(cs_1, :, t + 1), view(hs_2, :, t + 1), view(cs_2, :, t + 1), p, view(hs_1, :, t), view(cs_1, :, t), view(hs_2, :, t), view(cs_2, :, t), x)
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


function surrogate!(zs, hs_1, cs_1, hs_2, cs_2, p, xs, actions, weights)
    forward!(zs, hs_1, cs_1, hs_2, cs_2, p, xs)
    L = 0.0
    for t in eachindex(zs)
        a = actions[t]
        logp = a * logsigmoid(zs[t]) + (1 - a) * logsigmoid(-zs[t])
        L -= logp * weights[t]
    end
    return L
end


function trainLSTM(problem::ExcursionProblem, epochs::Int, LOG_INTERVAL::Int;
                  hidden = 16, batch = 32, lr = 0.01, clip = 1.0)
    @assert batch ≥ 2                           # baseline needs another traj to compare against
    T = problem.trajectory_length
    γ = problem.γ
    p = construct_LSTM(hidden)
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
            zs = zeros(T); hs_1 = zeros(hidden, T + 1); cs_1 = zeros(hidden, T + 1); hs_2 = zeros(hidden, T + 1); cs_2 = zeros(hidden, T + 1)
            dzs = zero(zs); dhs_1 = zero(hs_1); dcs_1 = zero(cs_1); dhs_2 = zero(hs_2); dcs_2 = zero(cs_2)
            Enzyme.autodiff(Reverse, surrogate!, Active,
                Duplicated(zs, dzs), Duplicated(hs_1, dhs_1), Duplicated(cs_1, dcs_1), Duplicated(hs_2, dhs_2), Duplicated(cs_2, dcs_2), Duplicated(p, dp),
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
    zs = zeros(T); hs_1 = zeros(length(view(p.V, 1, :)), T + 1); cs_1 = zeros(length(view(p.V, 1, :)), T + 1); hs_2 = zeros(length(view(p.V, 1, :)), T + 1); cs_2 = zeros(length(view(p.V, 1, :)), T + 1)
    forward!(zs, hs_1, cs_1, hs_2, cs_2, p, τ.xs)
    @assert zs ≈ τ.zs
end

function fd_check(problem, p; ε = 1e-6)         # Enzyme vs central finite differences
    τ = rollout(problem, p)
    T = problem.trajectory_length; n = length(view(p.V, 1, :))
    w = randn(T)
    f(q) = surrogate!(zeros(T), zeros(n, T + 1), zeros(n, T + 1), zeros(n, T + 1), zeros(n, T + 1), q, τ.xs, τ.actions, w)
    dp = Enzyme.make_zero(p)
    Enzyme.autodiff(Reverse, surrogate!, Active,
        Duplicated(zeros(T), zeros(T)), Duplicated(zeros(n, T + 1), zeros(n, T + 1)),
        Duplicated(zeros(n, T + 1), zeros(n, T + 1)), Duplicated(zeros(n, T + 1), zeros(n, T + 1)), Duplicated(zeros(n, T + 1), zeros(n, T + 1)), Duplicated(p, dp), Const(τ.xs), Const(τ.actions), Const(w))
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

function expand!(dict_hs_1, dict_cs_1, dict_hs_2, dict_cs_2, dict_ys, p, sequence, T)
    length(sequence) == T + 1 && return dict_hs_1, dict_cs_1, dict_hs_2, dict_cs_2, dict_ys    
    h_prev = dict_hs_1[sequence]
    c_prev = dict_cs_1[sequence]
    h_prev_2 = dict_hs_2[sequence]
    c_prev_2 = dict_cs_2[sequence]
    for x in (1, -1)
        push!(sequence, x)
        h_1 = zeros(length(view(p.V, 1, :)))
        c_1 = zeros(length(view(p.V, 1, :)))
        h_2 = zeros(length(view(p.V, 1, :)))
        c_2 = zeros(length(view(p.V, 1, :)))
        y = step!(h_1, c_1,h_2, c_2, p, h_prev,c_prev,h_prev_2, c_prev_2, x) #logit not probability
        dict_hs_1[copy(sequence)] = h_1
        dict_cs_1[copy(sequence)] = c_1
        dict_hs_2[copy(sequence)] = h_2
        dict_cs_2[copy(sequence)] = c_2
        dict_ys[copy(sequence)] = y
        expand!(dict_hs_1,dict_cs_1,dict_hs_2,dict_cs_2,dict_ys, p, sequence, T)
        pop!(sequence)                          # backtrack
    end
    return dict_hs_1,dict_cs_1,dict_hs_2,dict_cs_2,dict_ys
end

function build_states(p, h0, T)
    seq = [0]
    y0 = dot(view(p.V, 1, :), h0)
    dict_hs_1 = Dict{Vector{Int},typeof(h0)}(copy(seq) => h0)
    dict_cs_1 = Dict{Vector{Int},typeof(h0)}(copy(seq) => zeros(size(h0)))
    dict_hs_2 = Dict{Vector{Int},typeof(h0)}(copy(seq) => h0)
    dict_cs_2 = Dict{Vector{Int},typeof(h0)}(copy(seq) => zeros(size(h0)))
    dict_ys = Dict{Vector{Int},typeof(y0)}(copy(seq) => y0)
    return expand!(dict_hs_1,dict_cs_1,dict_hs_2,dict_cs_2,dict_ys, p, seq, T)
end

function LSTM_kl_divergence(problem::ExcursionProblem,policy, ys_dict)  #simple adaption from NN kl div
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
            exact_s = policy[s]

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
p0 = construct_LSTM(hidden_dim)
#check_replay(problem, p0)
#fd_check(problem, p0)
p = trainLSTM(problem, 200, 10)
hs_1, cs_1,hs_2,cs_2, ys = build_states(p,zeros(hidden_dim),T)
@load "data/solutions10-200.jld2" solutions
solution = solutions[T]
policy = Dict((k[1], k[2]) => v for (k, v) in solution.policy) #as it has a third key indexing T which is not needed here
Kl = LSTM_kl_divergence(problem,policy,ys)
println(Kl)

#rewards over time
#val state (0,0) == true returns
#opt policy - avg returns ≈ dkl, pseudolog scale 
