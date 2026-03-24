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
                label= i == 1 ? "Sampled (n=n_samples)" : nothing)
        end

        greedy_xs = greedy_trajectory_xs(greedy, problem)
        lines!(ax, collect(ts), greedy_xs, color=:red, linewidth=2.5, label="Greedy")

        axislegend(ax, unique=true)
        save("Rare_events.pdf",fig)
        fig
    end
    display(fig)
end

function plot_returns(solution::ExactSolution,epochs,tab_returns,nn_returns)
    expected_returns_plotter = solution.values[(0,0)].*ones(epochs)
    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="Epochs", ylabel="Rewards", title="Rewards",yscale = Makie.pseudolog10,xscale = log10)
        x_ax = 1:epochs

        lines!(ax, collect(x_ax), tab_returns, color=:red, linewidth=2.5, label="tabular returns")
                lines!(ax, collect(x_ax), nn_returns, color=:green, linewidth=2.5, label="neuralnet returns")
        lines!(ax,collect(x_ax), expected_returns_plotter, linestyle = :dash, label="theoretical max returns")
        axislegend(ax, position =:rb ,unique=true)  
        save("Rewards.pdf",fig)
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

function plot_kl_divergence(D_kl,LOG_INTERVAL,epochs)
    fig = begin
        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1,1], xlabel="Epochs", ylabel="D_kl", title="Evolution of KL divergence wrt to time",yscale = log10,xscale=log10)
        plot_epochs = 1:epochs/LOG_INTERVAL 

        lines!(ax, LOG_INTERVAL*collect(plot_epochs), D_kl, color=:red, linewidth=2.5,label = "Kl Divergence")

        # greedy_xs = greedy_trajectory_xs(greedy, problem)
        # lines!(ax, collect(ts), greedy_xs, color=:red, linewidth=2.5, label="Greedy")

        axislegend(ax, unique=true)
        save("log_D_kl.pdf",fig)
        fig
    end
    display(fig)
end