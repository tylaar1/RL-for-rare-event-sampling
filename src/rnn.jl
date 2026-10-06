using LinearAlgebra
using NNlib   

struct ExcursionProblem
    rewards::Array{Float64,3}
    trajectory_length::Int
    γ::Float64
end

function construct_RNN(input_dim::Int,hidden_dim::Int)
    a, b = -0.1, 0.1
    init(m, n) = rand(m, n) .* (b - a) .+ a

    V = init(input_dim, hidden_dim)    # hidden -> logit, requires input dim == output dim
    W = init(hidden_dim, hidden_dim)   # recurrent
    U = init(hidden_dim, input_dim)    # input -> hidden
    B = zeros(hidden_dim)

    return V,W,U,B
end

function forwards(V::Matrix{Float64},W::Matrix{Float64},U::Matrix{Float64},B::Vector{Float64},traj_length::Int64,init_state = 0.0,use_actuals=true)
    h_prev = zeros(length(B))
    hs = Vector{Float64}[]
    ys = Float64[]
    actuals = Int64[]
    x = init_state
    for _ in 1:traj_length
        h = tanh.(U * [x] .+ W * h_prev .+ B)
        push!(hs, h)
        y = only(sigmoid.(V * h))
        push!(ys, y)   # p_up
        a = sample_action(y)
        push!(actuals,a)
        if use_actuals
            x = a
        else
            x = y
        end
        h_prev = h
    end
    return hs,ys,actuals
end


function collect_rewards(env,ys,actions)
    state = 0
    rewards = Float64[]
    t=1
    for (y,a) in zip(ys,actions)
        p_action = a == 1 ? y : 1 - y
        state += 2*a - 1  #convert from 1/0 to 1/-1
        r = reward(env,(state,t,env.trajectory_length)) - log(p_action/0.5)
        push!(rewards,r)
        t += 1
    end
    
    return rewards
end

function sample_action(y)
    action = rand() < y ? 1 : 0
    return action   
end

function Loss(ys,actions,returns)
    up_log_probs = log.(clamp.(ys,0.0,1.0))
    down_log_probs = log.(clamp.(1.0 .- ys, 0.0, 1.0)) #clamp for numerical stability
    selected_log_probs = actions.* up_log_probs .+ (1.0 .- actions) .* down_log_probs #assuming actions is 0/1
    #selected_log_probs is the log probability assigned to the action taking place
    L_t = -(selected_log_probs .* returns)
    Loss = sum(L_t)
    return Loss
end

function diff_fwd(ys,actions)
    logit = log.(1 ./ ys .- 1)
end

function trainRNN(problem::ExcursionProblem,epochs::Int,LOG_INTERVAL::Int,args=false,solutions=false)
    V,W,U,B = construct_RNN(1,16) 
    λ = 0.01
    #opt = Adam(λ)
    #traj = Trajectory()
    for _ in 1:epochs
        hs, ys, actions = forwards(V,W,U,B,problem.trajectory_length)
        rewards = collect_rewards(problem,ys,actions)
        returns = reverse(cumsum(reverse(rewards)))
        for (name, v) in (("ys", ys), ("actions", actions), ("rewards", rewards), ("returns", returns))
            println(name, ": nan=", any(isnan, v), " inf=", any(isinf, v), " extrema=", extrema(v))
        end
        L = Loss(ys,actions,returns)
        println(actions,rewards,returns)
        println("--------------")
        println(L)
    end
    return nothing

end
# ------------------------------------------------------------------------
"from setup"
function reward(problem::ExcursionProblem, s′)
    x′, t′,T = s′
    problem.rewards[x′ + T + 1, t′,T]
end

function def_problem(T::Int64,bias::Float64,negative_penalty::Float64)
    #R = Random.randn(Float64, 2T+1, T)
    R = zeros( 2T+1, T,T) #now that we have arbitrary T we want to avoid learning noise
    R[1:T, :,T] .+= negative_penalty
    R[:, T,T] .= (-T:T) .^ 2 .* (-bias)
    return R
end

struct ExcursionProblem
    rewards::Array{Float64,3}
    trajectory_length::Int
    γ::Float64
end
#-------------------------------------------------------------------------
T = 10
bias = 5.0
negative_penalty = -10.0
γ = 1.0
R = def_problem(T, bias, negative_penalty)
problem = ExcursionProblem(R, T, γ)

trainRNN(problem,1,10)






function forward!(zs, hs, p, x0)
    x = x0                                     # [const] x0 is Const in autodiff
    for t in eachindex(zs)                     # [diff] the loop is unrolled: only operations are recorded
        hprev = view(hs, :, t)
        h     = view(hs, :, t + 1)
        mul!(h, p.W, hprev)                    # [diff] linear; gradient hits both W and h_{t-1}
        h .+= view(p.U, :, 1) .* x .+ p.B      # [diff] x depends on θ for t>1 (see feedback note below)
        h .= tanh.(h)                          # [diff] smooth; derivative 1 - h², saturates for |a| large
        z = dot(view(p.V, 1, :), h)            # [diff] linear
        zs[t] = z                              # [diff] mutation is fine for Enzyme, would break Zygote
        x = sigmoid(z)                         # [diff] open-loop feedback: gradient flows back through
    end                                        #        the input path, giving the extra σ'(z)·U·V term
    return nothing
end