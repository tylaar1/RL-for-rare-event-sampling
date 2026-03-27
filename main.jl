using CairoMakie
using ProgressBars
using Lux
using Random
using Optimisers
using Reactant
using Enzyme
using Statistics

include("types.jl")
include("problem_setup.jl")
include("tabular.jl")
include("neuralnet.jl")
include("plotters.jl")


function main()
    #setup env
    Random.seed!(1234)
    T = 10
    bias = 0.5 #should be a positive val
    negative_penalty = -1.0 #should be a negative val
    R = def_problem(T,bias,negative_penalty)
    γ = 1.0
    problem = ExcursionProblem(R, T, γ)

    #setup exact solution
    values = Dict{Tuple{Int64,Int64},Float64}()  
    policy = Dict{Tuple{Int64,Int64},Float64}() 
    solution = ExactSolution(values,policy)

    #calculate exact solution
    for s in state_space(problem)
        solution.values[s] = 0.0
        solution.policy[s] = 0.0
    end
    for s in reverse(collect(state_space(problem)))
        calculate_policy!(problem,solution,s)
    end

    #Set up tabular policy gradient
    params = Dict{Tuple{Int64,Int64},Float64}()  
    gradients = Dict{Tuple{Int64,Int64},Float64}()
    pga = PolicyGradient(γ, params, gradients)
    init_pga(pga, problem)

    #set up training hyperparams
    epochs = 1000
    batch_size = 64
    LOG_INTERVAL = 100
    α = 0.05

    #learn tabular policy
    tab_returns, D_kl = train!(pga, problem,solution, epochs, α, batch_size,LOG_INTERVAL)
    #learn NN policy
    pg_returns = trainPG(problem,epochs,batch_size)
    ac_returns = trainAC(problem,epochs,batch_size)
    """idea get Dkl for all methods to see how they coverge when plotted on same graph"""
    #***Comment/Uncomment plotting functions based on need***
    #plot_trajectories(pga,problem)
    plot_returns(solution,epochs,tab_returns,pg_returns,ac_returns)
    #plot_policy_comparison(pga,solution,problem)
    #plot_kl_divergence(D_kl,LOG_INTERVAL,epochs)
end
