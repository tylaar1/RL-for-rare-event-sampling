using CairoMakie
using ProgressMeter
using Lux
using Random
using Optimisers
using Enzyme
using Statistics
using LogExpFunctions
using CSV
using Tables
using Base.Threads
using JLD2
using DataFrames

include("types.jl")
include("problem_setup.jl")
include("tabular.jl")
include("neuralnet.jl")
include("plotters.jl")


function main()
    #setup env
    id = parse(Int,ARGS[1])
    T = 20 #used for tabular methods
    T_min_train = 10 
    T_max_train = 100
    T_array = [T for T in T_min_train:T_max_train if T % 2 == 0]
    bias = 5.0 #should be a positive val
    negative_penalty = -10.0 #should be a negative val
    R = def_problem(T,bias,negative_penalty)
    γ = 1.0
    problem = ExcursionProblem(R, T, γ)

    R_3D = def_3D_problem(T_min_train,T_max_train,bias,negative_penalty) #generates problem for series of T each equivelant to 2D version
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
        calculate_policy!(problem,solution,s)
    end

    #Set up tabular policy gradient
    params = Dict{Tuple{Int64,Int64,Int64},Float64}()  
    gradients = Dict{Tuple{Int64,Int64,Int64},Float64}()
    pga = PolicyGradient(γ, params, gradients)
    init_pga(pga, problem)

    #set up training hyperparams
    epochs = 1000
    batch_size = 64
    LOG_INTERVAL = 100
    α = 0.05
    #TODO: set up experiment with increments every 20 - include line at 100 for in sample and out sample results these should be good final results?? also fix args parsing so can control everything from these
    #learn tabular policy
    tab_returns, D_kl_tab = train!(pga, problem,solution, epochs, α, batch_size,LOG_INTERVAL)
    #learn NN policy
    pg_returns,D_kl_PG = trainPG(problem,problem_3D,epochs,batch_size,LOG_INTERVAL)
    #ac_returns,D_kl_AC = trainAC(problem,solution,epochs,batch_size,LOG_INTERVAL)
    #CSV.write("data/d_kl_$id.csv",(KL10 = D_kl_PG[:,1], KL12 = D_kl_PG[:,2],KL14 = D_kl_PG[:,3], KL16 = D_kl_PG[:,4],KL18 = D_kl_PG[:,5], KL20 = D_kl_PG[:,6]))
    T_step = parse(Int,ARGS[2])
    T_max_kl = parse(Int,ARGS[3])
    T_min_kl = parse(Int,ARGS[4])
    CSV.write("data/d_kl_$T_max_kl _$id.csv", NamedTuple(Symbol("KL$(T)") => D_kl_PG[:, i] for (i, T) in enumerate(T_min_kl:T_step:T_max_kl))) #2 = step size
    CSV.write("data/returns_$T_max_kl _$id.csv", (returns = vec(pg_returns),))

    #***Comment/Uncomment plotting functions based on need***
    #plot_trajectories(pga,problem)
    ac_returns = nothing
    D_kl_AC = nothing
    #plot_returns(solution,epochs,T,tab_returns,pg_returns,ac_returns)
    #plot_policy_comparison(pga,solution,problem)
    #plot_kl_divergence(LOG_INTERVAL,epochs,D_kl_tab,D_kl_PG,D_kl_AC)
    #plot_kl_divergence(20,1000,nothing,D_kl_PG)
    #plot_returns(epochs,pg_returns)
end

function get_exact_sols()
    #setup env
    Random.seed!(1234)
    T_min = 10
    T_max = 200
    T_array = [T for T in T_min:T_max if T % 2 == 0]
    bias = 5.0 #should be a positive val
    negative_penalty = -10.0 #should be a negative val
    γ = 1.0
    solutions = Dict{Int64,ExactSolution}()
    #setup exact solution
    for T in T_array
        R = def_problem(T, bias, negative_penalty)
        problem = ExcursionProblem(R, T, γ)
        values = Dict{Tuple{Int64,Int64,Int64}, Float64}()
        policy = Dict{Tuple{Int64,Int64,Int64}, Float64}()
        solution = ExactSolution(values, policy)
        for s in state_space(problem)
            solution.values[s] = 0.0
            solution.policy[s] = 0.0
        end
        for s in reverse(collect(state_space(problem)))
            calculate_policy!(problem, solution, s)
        end
    solutions[T] = solution
    println("solution for $T done")
    end
    @save "data/solutions10-200.jld2" solutions
end
    
function kl_plotter()
    paths = ["data/d_kl_200 _$i.csv" for i in 1:10]
    data_freq = 10 #only select every nth df so that graphs remain tidy
    means,std = prepair_data(paths,data_freq)

    plot_kl_divergence(100,1000,means)
    plot_kl_divergence_std(100,1000,means,std)
    plot_kl_div_final(paths)

end

function returns_plotter()
    #paths = ["data/returns_20 _$i.csv" for i in 1:10]
    paths = ["data/returns_200 _$i.csv" for i in 1:10]
    means,std = prepair_data_1D(paths)
    @load "data/solutions10-200.jld2" solutions
    T_min = 10 
    T_max = 100
    solutions_arr = [solutions[T].values[(0, 0, T)]/T for T in T_min:2:T_max]
    expected_max_return = mean(solutions_arr) #assume every T is will appear approx exquivelant amount of times over large amount of repeats
    plot_returns_std(1000,means,std,expected_max_return,"returns_normalised_std.pdf")
end


