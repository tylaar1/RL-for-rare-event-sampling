using LinearAlgebra
using NNlib   

input_dim, hidden_dim = 1, 16
a, b = -0.1, 0.1
init(m, n) = rand(m, n) .* (b - a) .+ a

V = init(input_dim, hidden_dim)    # hidden -> logit
W = init(hidden_dim, hidden_dim)   # recurrent
U = init(hidden_dim, input_dim)    # input -> hidden
B = zeros(hidden_dim)

X = [0.1, 0.2, 0.3]                
h_prev = zeros(hidden_dim)         
hs = Vector{Float64}[]
ys = Float64[]

function forwards(V,W,U,B,X)
    h_prev = zeros(length(B))
    hs = Vector{Float64}[]
    ys = Float64[]
    for x in X
        h = tanh.(U * [x] .+ W * h_prev .+ B)
        push!(hs, h)
        push!(ys, only(sigmoid.(V * h)))   # p_up
        h_prev = h
    end
    return hs,ys
end

"""
forwards will produce a vector y of actions, we then apply these sequentially to the env to collect rewards etc
"""

function collect_rewards(env,ys) #(pseudo_function)
    state = 0
    rewards = Float64[]
    for y in ys
        a = select_action(y)
        state += a
        r = env.get_reward(state)
        push!(rewards,r)
    end
    
    return rewards
end




forwards(V,W,U,B,X)

function Loss(ys,actions,returns)
    up_log_probs = log.(clamp.(ys,1f-7,1f0))
    down_log_probs = log.(clamp.(1f0 .- ys, 1f-7, 1f0)) #clamp for numerical stability
    selected_log_probs = actions.* up_log_probs .+ (1f0 .- actions) .* down_log_probs #assuming actions is 0/1
    #selected_log_probs is the log probability assigned to the action taking place
    L_t = selected_log_probs .* returns
    Loss = sum(L_t)
    return Loss
end



    