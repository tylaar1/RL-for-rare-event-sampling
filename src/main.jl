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
using ParameterSchedulers
using ColorSchemes
using LaTeXStrings
ENV["GKSwstype"] = "svg"
using Plots

include("types.jl")
include("problem_setup.jl")
include("tabular.jl")
include("neuralnet.jl")
include("plotters.jl")

function parse_args()
    return (
        id       = parse(Int, ARGS[2]),
        T_step   = parse(Int, ARGS[3]),
        T_max_kl = parse(Int, ARGS[4]),
        T_min_kl = parse(Int, ARGS[5]),
        Scale_inputs = parse(Int, ARGS[6]),
    )
end

function main()
    #setup env
    args = parse_args()
    T = 20 #used for tabular methods
    T_min_train = 100 
    T_max_train = 100
    T_array = [T for T in T_min_train:T_max_train if T % 2 == 0]
    bias = 5.0 #should be a positive val
    negative_penalty = -10.0 #should be a negative val
    R = def_problem(T,bias,negative_penalty)
    γ = 1.0
    problem = ExcursionProblem(R, T, γ)

    R_3D = def_3D_problem(T_min_train,T_max_train,bias,negative_penalty) #generates problem for series of T each equivelant to 2D version
    problem_3D = ExcursionProblem3D(R_3D,T_array,γ)

    #Set up tabular policy gradient
    params = Dict{Tuple{Int64,Int64,Int64},Float64}()  
    gradients = Dict{Tuple{Int64,Int64,Int64},Float64}()
    pga = PolicyGradient(γ, params, gradients)
    init_pga(pga, problem)

    #set up training hyperparams
    epochs = 10000
    batch_size = 64
    LOG_INTERVAL = 500
    α = 0.05
    @load "data/solutions10-200.jld2" solutions
    #learn tabular policy
    #println("start")
    #tab_returns, D_kl_tab = train!(pga, problem, epochs, α, batch_size,LOG_INTERVAL,solutions)
    #CSV.write("data/r20",(returns = vec(tab_returns),))
    #CSV.write("data/dklt20",(D_kl = vec(D_kl_tab),))
    #println("done")
    #learn NN policy
    @time((pg_returns,returns_unw,D_kl_PG) = trainPG(problem_3D,epochs,batch_size,LOG_INTERVAL,args,solutions))
    #ac_returns,D_kl_AC = trainAC(problem,solutions,epochs,batch_size,LOG_INTERVAL)
    #CSV.write("data/long/d_kl_$(args.T_max_kl) _$(args.id).csv", NamedTuple(Symbol("KL$(T)") => D_kl_PG[:, i] for (i, T) in enumerate(args.T_min_kl:args.T_step:args.T_max_kl))) #2 = step size
    #CSV.write("data/long/returns_$(args.T_max_kl) _$(args.id).csv", (returns = vec(pg_returns),))
    #CSV.write("data/long/returns_unw_$(args.T_max_kl) _$(args.id).csv", (returns = vec(returns_unw),))
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

function get_local_kl()
    #setup env
    Random.seed!(1234)
    T_min = 10
    T_max = 200
    T_array = [T for T in T_min:T_max if T % 2 == 0]
    bias = 5.0 #should be a positive val
    negative_penalty = -10.0 #should be a negative val
    γ = 1.0
    D_kl = []
    @load "data/solutions10-200.jld2" solutions
    #setup exact solution
    for T in T_array
        R = def_problem(T, bias, negative_penalty)
        problem = ExcursionProblem(R, T, γ)
        values = Dict{Tuple{Int64,Int64,Int64}, Float64}()
        policy = Dict{Tuple{Int64,Int64,Int64}, Float64}()
        local_solution = ExactSolution(values, policy)
        for s in state_space(problem)
            local_solution.values[s] = 0.0
            local_solution.policy[s] = 0.0
        end
        for s in reverse(collect(state_space(problem)))
            calculate_local_policy!(problem, local_solution, s)
        end
        push!(D_kl, efficient_kl(problem,local_solution,solutions[T]))
    end
    return D_kl
end
    
function kl_plotter()
    paths = ["data/long/d_kl_200 _$i.csv" for i in 1:10]
    data_freq = 10 #only select every nth df so that graphs remain tidy
    means,std = prepair_data(paths,data_freq)
    data_log_int = 500
    data_epochs = 50000
    save_type = "svg"
    #plot_kl_divergence_std(data_log_int,data_epochs,means,std)
    normalise = [true,false,"unbiased"]
    local_kl=true
    plot_kl_divergence(data_log_int,data_epochs,means,save_type,"to_origin")
    for arg in normalise
        plot_kl_div_final(paths,arg,save_type,local_kl)
        #plot_kl_divergence(data_log_int,data_epochs,means,save_type,"to_origin")
    end
end

function returns_plotter()
    paths = ["data/long/returns_200 _$i.csv" for i in 1:10]
    means,std = prepair_data_1D(paths)
    @load "data/solutions10-200.jld2" solutions
    T_min = 10 
    T_max = 100
    data_epochs = 50000
    solutions_arr = [solutions[T].values[(0, 0, T)]/T for T in T_min:2:T_max]
    expected_max_return = mean(solutions_arr) #assume every T is will appear approx exquivelant amount of times over large amount of repeats
    plot_returns_std(data_epochs,means,std,expected_max_return,"returns_normalised_std.pdf")
end

function simple_plot()
    r20    = CSV.read("data/r20.csv",    DataFrame)
    dklt20 = CSV.read("data/dklt20.csv", DataFrame)
    r20 = r20.returns #size = 50000
    stop = length(r20)
    n=length(r20) ÷ 100 #n = 500
    r20 = [mean(r20[i:i+n-1]) for i in 1:n:length(r20)-n+1]
    step = stop/length(r20) # = 100
    dklt20 = dklt20.D_kl # size = 100
    fig = CairoMakie.Figure(size=(900, 400))
    @load "data/solutions10-200.jld2" solutions
    s = solutions[20].values[(0,0,20)]
    ax1 = CairoMakie.Axis(fig[1,1],
        xlabel="Epochs", ylabel="Returns",
        title="Returns for T = 20",
        yscale=Makie.pseudolog10,
        xscale=log10)

    ax2 = CairoMakie.Axis(fig[1,2],
        xlabel="Epochs", ylabel=L"D_{KL}", 
        title="KL divergence for T = 20",
        yscale=Makie.pseudolog10,
        xscale=log10)

    CairoMakie.lines!(ax1, 1:step:stop,   r20,label="Sampled Returns")
    CairoMakie.lines!(ax1,1:step:stop, s*ones(length(1:step:stop)), color= :red, linestyle = :dash, label="Theoretical max returns")
    CairoMakie.lines!(ax2, 1:step:stop, dklt20)
    CairoMakie.save("r_is_kl.svg", fig)
end

function unbiased_kl()
    T_min = 10
    T_max = 200
    T_array = [T for T in T_min:T_max if T % 2 == 0]
    kls = Dict{Int64,Float64}()
    @load "data/solutions10-200.jld2" solutions
    bias = 5.0 #should be a positive val
    negative_penalty = -10.0 #should be a negative val
    γ = 1.0
    for T in T_array
        R = def_problem(T,bias,negative_penalty)
        problem = ExcursionProblem(R, T, γ)
        params = Dict{Tuple{Int64,Int64,Int64},Float64}()  
        gradients = Dict{Tuple{Int64,Int64,Int64},Float64}()
        pga = PolicyGradient(γ, params, gradients)
        for s in state_space(problem)
            pga._policy_parameters[s]=0.5
        end
        unbiased_kl = efficient_kl(problem,pga,solutions[T])
        kls[T] = unbiased_kl
    end
    @save "data/unbiased10-200.jld2" kls
    return kls
end

