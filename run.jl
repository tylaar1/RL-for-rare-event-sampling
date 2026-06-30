include("src/main.jl")

const COMMANDS = Dict(
    "train"        => main,
    "exact_sols"   => get_exact_sols,
    "plot_returns" => returns_plotter,
    "plot_kl"      => kl_plotter,
    "plot_all"     => () -> (returns_plotter(); kl_plotter()),
    "simple_plot"  => simple_plot,
    "unbiased_kl"  => unbiased_kl,
    "plot_visitation" => plot_visitation_density
)

function run()
    if isempty(ARGS)
        println("Usage: julia run.jl <command> [args...]")
        println("Commands: $(join(keys(COMMANDS), ", "))")
        return
    end
    
    cmd = ARGS[1]
    if haskey(COMMANDS, cmd)
        COMMANDS[cmd]()
    else
        println("Unknown command: $cmd. Available: $(join(keys(COMMANDS), ", "))")
    end
end

run()