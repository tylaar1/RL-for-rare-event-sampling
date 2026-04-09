using CairoMakie
using ProgressMeter
using Lux
using Random
using Optimisers
using Enzyme
using Statistics
using LogExpFunctions

include("types.jl")
include("problem_setup.jl")
include("tabular.jl")
include("neuralnet.jl")
include("plotters.jl")


function main()
    #setup env
    Random.seed!(1234)
    T = 14 #arbitrary value within T min and T max to calculate kl against.. to be improved on later
    T_min = 10
    T_max = 20
    T_array = [T for T in T_min:T_max if T % 2 == 0]
    bias = 5.0 #should be a positive val
    negative_penalty = -10.0 #should be a negative val
    R = def_problem(T,bias,negative_penalty)
    γ = 1.0
    problem = ExcursionProblem(R, T, γ)

    R_3D = def_3D_problem(T_min,T_max,bias,negative_penalty) #generates problem for series of T each equivelant to 2D version
    problem_3D = ExcursionProblem3D(R_3D,T_array,γ)

    #setup exact solution
    values = Dict{Tuple{Int64,Int64,Int64},Float64}()  
    policy = Dict{Tuple{Int64,Int64,Int64},Float64}() 
    solution = ExactSolution(values,policy)
    #calculate exact solution
    
    for s in state_space(problem) 
        solution.values[s] = 0.0
        solution.policy[s] = 0.0
    end
    for s in reverse(collect(state_space(problem)))
        calculate_policy!(problem,solution,s) #TODO:save and load in results for these for each of T_array
    end

    #Set up tabular policy gradient
    params = Dict{Tuple{Int64,Int64,Int64},Float64}()  
    gradients = Dict{Tuple{Int64,Int64,Int64},Float64}()
    pga = PolicyGradient(γ, params, gradients)
    init_pga(pga, problem)

    #set up training hyperparams
    epochs = 100
    batch_size = 64
    LOG_INTERVAL = 10
    α = 0.05

    #learn tabular policy
    tab_returns, D_kl_tab = train!(pga, problem,solution, epochs, α, batch_size,LOG_INTERVAL)
    #learn NN policy
    pg_returns,D_kl_PG = trainPG(problem,problem_3D,solution,epochs,batch_size,LOG_INTERVAL)
    #ac_returns,D_kl_AC = trainAC(problem,solution,epochs,batch_size,LOG_INTERVAL)
    
    #***Comment/Uncomment plotting functions based on need***
    #plot_trajectories(pga,problem)
    ac_returns = nothing
    D_kl_AC = nothing
    plot_returns(solution,epochs,T,tab_returns,pg_returns,ac_returns)
    #plot_policy_comparison(pga,solution,problem)
    plot_kl_divergence(LOG_INTERVAL,epochs,D_kl_tab,D_kl_PG,D_kl_AC)
end
