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
        local_solution.values[s] = 0.0
        local_solution.policy[s] = 0.0
    end
    for s in reverse(collect(state_space(problem)))
        calculate_local_policy!(problem,solution, s)
    end
    for s in state_space(problem)
        x,t,T = s
        i = x + T + 1                     # row for position x
        p = probs[i, t + 1]               # mass currently at (x, t)
        p == 0.0 && continue
        p_up = solution.policy[s]
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
function plot_visitation_density(
    T::Int;
    title::AbstractString          = "State Visitation",
    x_min                          = nothing,
    x_max                          = nothing,
    colormap::Symbol               = :turbo,
    colorbar_title::AbstractString = "P(visit)",
    smooth_gaps::Bool              = true,
    savepath::AbstractString       = "",
)
    probs = compute_visitation(T)

    if x_min === nothing || x_max === nothing
        x_min, x_max = _auto_window(probs, T)
    end
    x_range = x_min:x_max

    # Display matrix: columns are times t = 0 .. T-1 (terminal column dropped,
    # matching the example figures). Unreachable cells stay NaN -> blank.
    visit_map = fill(NaN, length(x_range), T)
    for t in 0:(T - 1), x in x_range
        abs(x) > t && continue                       # outside triangle: leave NaN
        visit_map[x - x_min + 1, t + 1] = probs[x + T + 1, t + 1]
    end

    max_val = maximum(filter(!isnan, visit_map); init = 1.0)
    smooth_gaps && _fill_nans!(visit_map, _reachable_gap_mask(x_range, T))

    p = heatmap(
        0:(T - 1), x_range, visit_map;
        xlabel         = "Time Step",
        ylabel         = "State (Position)",
        title          = title,
        colorbar_title = colorbar_title,
        clims          = (0.0, max_val),
        color          = colormap,
    )

    isempty(savepath) || savefig(p, savepath)
    return p
end

# ---------------------------------------------------------------------
# 3. PLUGGING IN YOUR OWN POLICY
# ---------------------------------------------------------------------
#
# `policy` is just `(x, t) -> (p_down, p_up)`. If you have a solved
# algorithm `alg` and `problem` from the NeuralExcursion package, wrap its
# `action_probabilities` like this and pass `pol` to plot_visitation_density:
#
#     using NeuralExcursion
#     pol(x, t) = action_probabilities(alg, problem, (x, t))   # -> (p_down, p_up)
#     plot_visitation_density(pol, horizon(problem);
#                             title = "State Visitation — ERBI Exact Policy")
#
# Anything callable with that signature works — a neural net, a tabular
# policy, an analytic formula, etc.

# ---------------------------------------------------------------------
# 4. RUNNABLE DEMO (no external code needed)
# ---------------------------------------------------------------------
#
# A Doob h-transform turning the simple symmetric walk into a discrete
# *excursion*: a bridge pinned back to 0 at time T and conditioned to stay
# positive. This is just here so the file runs on its own and shows the
# same "arch" shape as the example figures; swap in your own policy above.

"""
    excursion_policy(T) -> (policy, h)

Returns a `policy(x, t)` for the simple-random-walk excursion on `0:T`
(bridge from 0 to 0, conditioned to stay ≥ 0), built by Doob's
h-transform. `h[x+1, t+1]` is the (unnormalised) number of positive paths
from `(x, t)` to `(0, T)`; the up-probability is `h(x+1,t+1)/(2 h(x,t))`.
"""
function excursion_policy(T::Int)
    # h[x+1, t+1] = # of nonneg-staying paths from (x,t) to (0,T).
    h = zeros(Float64, T + 2, T + 1)
    h[1, T + 1] = 1.0                                  # at t=T only x=0 counts
    for t in (T - 1):-1:0, x in 0:t
        up   = (x + 1 <= T)      ? h[x + 2, t + 2] : 0.0
        down = (x - 1 >= 0)      ? h[x,     t + 2] : 0.0
        h[x + 1, t + 1] = up + down
    end
    function policy(x, t)
        x < 0 && return (1.0, 0.0)                     # should not happen for an excursion
        denom = h[x + 1, t + 1]
        denom == 0.0 && return (0.5, 0.5)              # unreachable; value is irrelevant
        p_up   = (x + 1 <= T) ? h[x + 2, t + 2] / denom : 0.0
        p_up   = clamp(p_up, 0.0, 1.0)
        return (1.0 - p_up, p_up)
    end
    return policy, h
end

# Execute the demo only when this file is run directly as a script.
if abspath(PROGRAM_FILE) == @__FILE__
    T = 100
    policy, _ = excursion_policy(T)
    p = plot_visitation_density(
        policy, T;
        title    = "State Visitation — Excursion (demo)",
        savepath = "visitation_demo.svg",
    )
    display(p)
    println("Saved visitation_demo.svg")
end
