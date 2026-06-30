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

# =====================================================================
#  visitation_density_standalone.jl
#
#  Self-contained recipe for the "State Visitation" density heatmaps
#  (e.g. "State Visitation — ERBI Exact Policy").
#
#  This file has NO dependency on the NeuralExcursion package. The only
#  thing it needs from *your* code is a policy: a function
#
#       policy(x, t) -> (p_down, p_up)
#
#  giving the probability of stepping down (to x-1) or up (to x+1) from
#  lattice site x at time t, with p_down + p_up = 1.
#
#  The model is a ±1 walk on the integers starting at x = 0, t = 0:
#      "up"   :  (x, t) -> (x+1, t+1)   with prob p_up
#      "down" :  (x, t) -> (x-1, t+1)   with prob p_down
#
#  Two pieces of maths live here:
#    1. compute_visitation : forward-propagate the start mass through the
#       lattice to get P(X_t = x) for every (x, t).   <-- "the maths"
#    2. plot_visitation_density : crop to a display window, smooth the
#       parity gaps, and draw the heatmap.            <-- "the graphing"
#
#  Run `julia visitation_density_standalone.jl` to see the demo at the
#  bottom, or `include` this file and call plot_visitation_density(...).
# =====================================================================

using Plots

# ---------------------------------------------------------------------
# 1. THE MATHS: policy  ->  state-visitation density
# ---------------------------------------------------------------------

"""
    compute_visitation(policy, T) -> Matrix{Float64}

Forward pass through the lattice. Returns a `(2T+1, T+1)` matrix `probs`
where

    probs[x + T + 1, t + 1] == P(X_t = x)

i.e. the probability that the walk is at position `x` at time `t` under
`policy`. Row index `x + T + 1` maps integer position `x ∈ -T:T` to a
1-based row; column index `t + 1` maps time `t ∈ 0:T`.

`policy` must be callable as `policy(x, t) -> (p_down, p_up)`.

How it works: all mass starts on the origin cell `(x=0, t=0)`. We sweep
forward in time; from each occupied cell we push its mass to the two
children, weighted by the policy's down/up probabilities. Because the
walk steps ±1, only sites with the same parity as `t` and `|x| ≤ t` are
ever reachable — that is the `x in -t:2:t` inner loop.
"""
function compute_visitation(T::Int)
    bias = 5.0 #should be a positive val
    negative_penalty = -10.0 #should be a negative val
    γ = 1.0
    probs = zeros(Float64, 2T + 1, T + 1)
    probs[T + 1, 1] = 1.0                 # start: P(X_0 = 0) = 1
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
        calculate_local_policy!(problem,solution, s)
    end
    for s in state_space(problem)
        x,t,T = s
        i = x + T + 1                     # row for position x
        p = probs[i, t + 1]               # mass currently at (x, t)
        p == 0.0 && continue
        p_up = sigmoid(solution.policy[s])
        p_down = 1 - p_up
        probs[i + 1, t + 2] += p * p_up   # mass flowing to (x+1, t+1)
        probs[i - 1, t + 2] += p * p_down # mass flowing to (x-1, t+1)
    end
    return probs
end

# ---------------------------------------------------------------------
# 2. THE GRAPHING: density matrix  ->  heatmap
# ---------------------------------------------------------------------

"""
    _reachable_gap_mask(x_range, T) -> BitMatrix

Marks the "gap" cells of the display matrix: sites that lie *inside* the
reachable triangle (`|x| ≤ t`) but have the wrong parity for time `t`
(`isodd(x - t)`), so they are never visited yet sit between valid lattice
points. These produce the checkerboard of blanks; we want to smear colour
into them so the heatmap reads as a continuous density. Cells *outside*
the triangle (`|x| > t`) are left unmarked — they stay blank and form the
clean triangular boundary you see at the left of the figure.
"""
function _reachable_gap_mask(x_range::UnitRange{Int}, T::Int)
    gaps = falses(length(x_range), T)
    for t in 0:(T - 1), x in x_range
        abs(x) > t   && continue          # outside the reachable triangle
        isodd(x - t) || continue          # same parity => real lattice point
        gaps[x - first(x_range) + 1, t + 1] = true
    end
    return gaps
end

"""
    _fill_nans!(mat, mask)

Iteratively replaces `NaN` entries of `mat` that lie inside `mask` with
the average of their non-NaN 4-neighbours, repeating until nothing more
changes. This visually fills the parity gaps without touching the
genuinely-unreachable cells outside the triangle (which are not in
`mask`).
"""
function _fill_nans!(mat::AbstractMatrix{Float64}, mask::BitMatrix)
    rows, cols = size(mat)
    changed = true
    while changed
        changed = false
        for c in 1:cols, r in 1:rows
            (mask[r, c] && isnan(mat[r, c])) || continue
            s = 0.0; n = 0
            r > 1    && !isnan(mat[r-1, c]) && (s += mat[r-1, c]; n += 1)
            r < rows && !isnan(mat[r+1, c]) && (s += mat[r+1, c]; n += 1)
            c > 1    && !isnan(mat[r, c-1]) && (s += mat[r, c-1]; n += 1)
            c < cols && !isnan(mat[r, c+1]) && (s += mat[r, c+1]; n += 1)
            n > 0 && (mat[r, c] = s / n; changed = true)
        end
    end
end

"""
    _auto_window(probs, T; threshold, pad) -> (x_min, x_max)

Picks a symmetric-ish display window in `x` by finding the smallest and
largest positions that ever carry visitation mass above `threshold`,
padded by `pad`. This reproduces the tight y-axis of the example figures
(roughly -20..20) automatically instead of showing the whole -T..T range.
"""
function _auto_window(probs::Matrix{Float64}, T::Int; threshold = 1e-6, pad = 2)
    occupied = vec(any(>(threshold), probs; dims = 2))   # which rows ever lit
    rows = findall(occupied)
    isempty(rows) && return (-1, 1)
    x_lo = (first(rows) - 1) - T          # row r corresponds to x = r-1-T
    x_hi = (last(rows)  - 1) - T
    return (max(x_lo - pad, -T), min(x_hi + pad, T))
end

"""
    plot_visitation_density(policy, T; kwargs...) -> Plots.Plot

Builds the state-visitation density under `policy` over horizon `T` and
draws it as a heatmap (time on the x-axis, position on the y-axis,
P(visit) as colour).

Keyword arguments:
  title          : plot title.
  x_min, x_max   : position display window. Leave as `nothing` to auto-crop
                   to the region that actually carries mass.
  colormap       : colour scheme (`:turbo` matches the example figures;
                   `:plasma`, `:viridis`, ... also work).
  colorbar_title : label on the colour bar.
  smooth_gaps    : fill the parity checkerboard for a continuous look.
  savepath       : if non-empty, save the figure there (e.g. "vis.svg").
"""
function plot_visitation_density()
    T = 40
    title = "State Visitation"
    x_min = -40
    x_max = 40
    colormap = :turbo
    colorbar_title = "P(Visit)"
    smooth_gaps = true
    savepath = "one_step.svg"

    probs = compute_visitation(T)

    if x_min === nothing || x_max === nothing
        x_min, x_max = _auto_window(probs, T)
    end
    x_range = x_min:x_max

    # Display matrix: columns are times t = 0 .. T-1 (terminal column dropped,
    # matching the example figures). Unreachable cells stay NaN -> blank.
    visit_map = fill(NaN, length(x_range), T)
    for t in 0:(T - 1), x in x_range
        abs(x) > t && continue    
        iseven(x - t) || continue             
        visit_map[x - x_min + 1, t + 1] = probs[x + T + 1, t + 1]
    end
    #visit_map = log.(visit_map)
    max_val = maximum(filter(!isnan, visit_map); init = 1.0)
    println(max_val)
    smooth_gaps && _fill_nans!(visit_map, _reachable_gap_mask(x_range, T))

    p = Plots.heatmap(
        0:(T - 1), x_range, visit_map;
        xlabel         = "Time Step",
        ylabel         = "State (Position)",
        title          = title,
        colorbar_title = colorbar_title,
        clims          = (0.0,1.0),
        color          = colormap,
    )

    isempty(savepath) || savefig(p, savepath)
    return p
end

