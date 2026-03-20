#using CairoMakie
#using ProgressBars
using Lux
using Random
using Optimisers
using Reactant
using Enzyme

#include("plotters.jl") #TODO fix load order to avoid undef var error 
function train(epochs::Int);
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    Random.TaskLocalRNG()

    # Construct the layer
    model = Chain(Dense(2, 64, tanh), Chain(Dense(64, 64, tanh), Chain(Dense(64, 32, tanh), Dense(32, 1, sigmoid)))) #input state,output action probs

    # Get the device determined by Lux
    dev = reactant_device()

    # Parameter and State Variables
    ps, st = Lux.setup(rng, model) |> dev

    # Dummy Input
    x = rand(rng, Float32, 2, 200) |> dev

    # Run the model
    ## We need to use @jit to compile and run the model with Reactant
    y, st = @jit Lux.apply(model, x, ps, st)

    ## For best performance, first compile the model with Reactant and then run it
    apply_compiled = @compile Lux.apply(model, x, ps, st)
    apply_compiled(model, x, ps, st)

    # Gradients
    ## First construct a TrainState
    train_state = Training.TrainState(model, ps, st, Adam(0.0001f0))

    losses = Float32[]
    for _ in 1:epochs
    # Both these steps can be combined into a single call (preferred approach)
        gs, loss, stats, train_state = Training.single_train_step!(
            AutoEnzyme(), #auto diff
            MSELoss(), #loss function
            (x, dev(rand(rng, Float32, 1, 200))), #input/target pair
            train_state #model params etc
        )

        push!(losses, Float32(loss))
    end
    return  losses
end


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


function main()
    Random.seed!(1234)
    T = 10
    bias = 0.5
    R = Random.randn(Float64, 2T+1, T)
    #R = zeros( 2T+1, T)
    R[1:T, :] .-= 1
    R[:, T] .= (-T:T) .^ 2 .* (-bias)

    γ = 1.0
    α = 0.1

    values = Dict{Tuple{Int64,Int64},Float64}()  
    policy = Dict{Tuple{Int64,Int64},Float64}() 

    problem = ExcursionProblem(R, T, γ)
    params = Dict{Tuple{Int64,Int64},Float64}()  
    gradients = Dict{Tuple{Int64,Int64},Float64}()


    epochs = 1000
    batch_size = 64
    LOG_INTERVAL = 100
    avg_returns, D_kl = train!(pga, problem,solution, epochs, α, batch_size,LOG_INTERVAL)

end

