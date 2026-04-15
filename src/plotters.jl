using CairoMakie


function plot_trajectories(pga::PolicyGradient,problem::ExcursionProblem)
    greedy = greedy_policy(pga)
    n_samples = 30
    T = problem.trajectory_length
    ts = 0:T
    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="t", ylabel="x", title="Rare Event Trajectories")

        for i in 1:n_samples
            xs = sampled_trajectory_xs(pga, problem)
            lines!(ax, collect(ts), xs, color=(:blue, 0.20), linewidth=1,
                label = i == 1 ? "Sampled (n=n_samples)" : nothing)
        end

        greedy_xs = greedy_trajectory_xs(greedy, problem)
        lines!(ax, collect(ts), greedy_xs, color=:red, linewidth=2.5, label="Greedy")

        axislegend(ax, unique=true)
        save("Rare_events.pdf",fig)
        fig
    end
    display(fig)
end

function plot_returns(solution::ExactSolution,epochs,T,tab_returns=nothing,pg_returns=nothing,ac_returns=nothing)
    expected_returns_plotter = solution.values[(0,0,T)].*ones(epochs)
    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="Epochs", ylabel="Rewards", title="Rewards",yscale = Makie.pseudolog10,xscale = log10)
        x_ax = 1:epochs

        if tab_returns !== nothing
            lines!(ax, collect(x_ax), tab_returns, color=:red, linewidth=2.5, label="Tabular returns")
        end
        if pg_returns !== nothing
            lines!(ax, collect(x_ax), pg_returns, color=:green, linewidth=2.5, label="PolicyGradient returns")
        end
        if ac_returns !== nothing
             lines!(ax, collect(x_ax), ac_returns, color=:purple, linewidth=2.5, label="ActorCritic returns")
        end
        lines!(ax,collect(x_ax), expected_returns_plotter, linestyle = :dash, label="Theoretical max returns")
        axislegend(ax, position =:rb ,unique=true)  
        save("Rewards.pdf",fig)
        fig
    end
    display(fig)    
end

function plot_returns_std(epochs,pg_returns,pg_returns_std,expected_returns,saveas=nothing)
    expected_returns_plotter = expected_returns*ones(epochs)
    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="Epochs", ylabel="Returns/T", title="Returns/T",yscale = Makie.pseudolog10,xscale = log10)
        x_ax = 1:epochs
        upper = pg_returns .+ pg_returns_std
        lower = pg_returns .- pg_returns_std
        band!(ax, x_ax, lower, upper)
        lines!(ax, collect(x_ax), pg_returns, linewidth=2.5, label="PolicyGradient returns")
        lines!(ax,collect(x_ax), expected_returns_plotter, linestyle = :dash, label="Theoretical max returns")
        axislegend(ax, position =:rb ,unique=true) 
        if saveas == nothing 
            save("Rewards_std.pdf",fig)
        else
            save(saveas,fig)
        end
        fig
    end
    display(fig)    
end


function plot_policy_comparison(pga::PolicyGradient, solution::ExactSolution, problem::ExcursionProblem)
    T = problem.trajectory_length
    exact_matrix   = fill(NaN, 2T+1, T)
    learned_matrix = fill(NaN, 2T+1, T)
    diff_matrix    = fill(NaN, 2T+1, T)

    for (s, p_exact) in solution.policy
        x, t = s
        if t == 0 continue end  # skip t=0 for cleaner plot, only one state
        p_learned = sigmoid(pga._policy_parameters[s])
        row = x + T + 1
        exact_matrix[row, t]   = p_exact
        learned_matrix[row, t] = p_learned
        diff_matrix[row, t]    = p_learned - p_exact
    end

    fig = CairoMakie.Figure(size=(1400, 500))

    ax1 = CairoMakie.Axis(fig[1,1], xlabel="t", ylabel="x", title="Exact Policy (p_up)")
    hm1 = heatmap!(ax1, 1:T, -T:T, exact_matrix', colormap=:RdBu, colorrange=(0,1))
    Colorbar(fig[1,2], hm1)

    ax2 = CairoMakie.Axis(fig[1,3], xlabel="t", ylabel="x", title="Learned Policy (p_up)")
    hm2 = heatmap!(ax2, 1:T, -T:T, learned_matrix', colormap=:RdBu, colorrange=(0,1))
    Colorbar(fig[1,4], hm2)

    ax3 = CairoMakie.Axis(fig[1,5], xlabel="t", ylabel="x", title="Difference (Learned - Exact)")
    hm3 = heatmap!(ax3, 1:T, -T:T, diff_matrix', colormap=:RdBu, colorrange=(-1,1))
    Colorbar(fig[1,6], hm3)

    save("policy_comparison.pdf", fig)
    display(fig)
    return fig
end

function plot_kl_divergence(LOG_INTERVAL,epochs,D_kl=nothing,pg=nothing,ac=nothing)
    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="Epochs", ylabel="D_kl", title="Evolution of KL divergence wrt to time",yscale = log10,xscale=log10)
        plot_epochs = 1:epochs/LOG_INTERVAL 
        #if D_kl !== nothing
        #    lines!(ax, LOG_INTERVAL*collect(plot_epochs), D_kl, color=:red, linewidth=2.5,label = "Tabular KL Divergence")
        #end
        if pg !== nothing
            pg = Matrix(pg)
            _,N = size(pg)
            for i in 1:N
                lines!(ax, LOG_INTERVAL*collect(plot_epochs), pg[:,i], linewidth=2.5,label = "Policy Gradient KL Divergence $(8+2*i)") #8+2*i only works for numbers starting at 10 and increments of 2
            end
        end
        if ac !== nothing
            lines!(ax, LOG_INTERVAL*collect(plot_epochs), ac, color=:purple, linewidth=2.5,label = "Actor Critic KL Divergence")
        end
        axislegend(ax, unique=true)
        save("log_D_kl.pdf",fig)
        fig
    end
    display(fig)
end

function plot_kl_divergence(LOG_INTERVAL,epochs,pg::DataFrame)
    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="Epochs", ylabel="D_kl", title="Evolution of KL divergence wrt to time",yscale = log10,xscale=log10)
        plot_epochs = 1:epochs/LOG_INTERVAL 
        if pg !== nothing
            pg = Matrix(pg)
            _,N = size(pg)
            for i in 1:N
                lines!(ax, LOG_INTERVAL*collect(plot_epochs), pg[:,i], linewidth=2.5,label = "Policy Gradient KL Divergence $(8+2*i)") #8+2*i only works for numbers starting at 10 and increments of 2
            end
        end
        axislegend(ax, unique=true)
        save("log_D_kl_avg.pdf",fig)
        fig
    end
    display(fig)
end

function prepair_data(filepaths::Vector{String}, std_dev=true)
    dfs = [CSV.read(f, DataFrame) for f in filepaths]
    cols = names(dfs[1])

    data = cat([Matrix(df) for df in dfs]..., dims=3)
    
    means = DataFrame(mean(data, dims=3)[:,:,1], cols)
    
    if std_dev
        stds = DataFrame(std(data, dims=3)[:,:,1], cols)
        return means, stds
    else
        return means
    end
end

function prepair_data_1D(filepaths::Vector{String}, std_dev=true)
    dfs = [CSV.read(f, DataFrame) for f in filepaths]
    
    data = hcat([Vector(df[:, 1]) for df in dfs]...)
    
    means = vec(mean(data, dims=2))
    
    if std_dev
        stds = vec(std(data, dims=2))
        return means, stds
    else
        return means
    end
end

function plot_kl_divergence_std(LOG_INTERVAL, epochs, pg=nothing, pg_std=nothing)

    pg  = Matrix(pg)
    _, N = size(pg)

    # --- auto-fit grid dimensions ---
    ncols = ceil(Int, sqrt(N))
    nrows = ceil(Int, N / ncols)

    plot_epochs = LOG_INTERVAL .* collect(1:size(pg, 1))

    fig = CairoMakie.Figure(size=(400 * ncols, 350 * nrows))

    for i in 1:N
        row = ceil(Int, i / ncols)
        col = mod1(i, ncols)

        label_val = 8 + 2 * i   # 10, 12, 14, ...
        ax = CairoMakie.Axis(
            fig[row, col],
            xlabel  = "Epochs",
            ylabel  = "D_kl",
            title   = "KL Divergence — Length $(label_val)",
            yscale  = log10,
            xscale  = log10,
        )

        color = Makie.wong_colors()[mod1(i, length(Makie.wong_colors()))]

        # shaded std dev band
        if pg_std !== nothing
            pg_std_mat = Matrix(pg_std)
            lower = max.(pg[:, i] .- pg_std_mat[:, i], 1e-12)   # clamp away from ≤0 for log scale
            upper = pg[:, i] .+ pg_std_mat[:, i]
            CairoMakie.band!(ax, plot_epochs, lower, upper,
                color = (color, 0.25))
        end

        CairoMakie.lines!(ax, plot_epochs, pg[:, i],
            color     = color,
            linewidth = 2.5,
            label     = "PG KL Divergence $(label_val)")
    end

    # hide any leftover empty cells in the grid
    for i in (N+1):(nrows * ncols)
        row = ceil(Int, i / ncols)
        col = mod1(i, ncols)
        CairoMakie.hidedecorations!(CairoMakie.Axis(fig[row, col]))
    end

    CairoMakie.save("log_D_kl_std.pdf", fig)
    display(fig)
end

