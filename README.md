# Rare Trajectory Sampling via Reinforcement Learning

Sampling rare trajectories from stochastic systems is a fundamental challenge in statistical physics and applied probability. This project frames the problem as a reinforcement learning task: train a policy to guide a discrete random walker toward low-probability excursions, without enumerating the full trajectory space.

## Problem

A random walker starts at the origin of a discrete 1D lattice and takes steps of ±1. The goal is to sample atypical trajectories — those that stray far from the origin and return — which occur with exponentially small probability under the natural unbiased dynamics. A reward signal biased toward large excursions is defined over the state space `(x, t, T)`, where `x` is position, `t` is time, and `T` is trajectory length.

Naive Monte Carlo is ineffective here. RL provides a principled way to bias the walker toward rare events, with KL regularisation ensuring the learned policy doesn't collapse to a deterministic path.

## Methods

Three RL approaches are implemented and compared:

- **Policy Iteration (Exact)** — dynamic programming over the full state space; provides ground-truth optimal policy and value function for validation
- **REINFORCE (Tabular)** — policy gradient with a tabular sigmoid-parameterised policy; trained via Monte Carlo returns
- **REINFORCE (Neural Network)** — policy gradient with a 4-layer MLP (Lux.jl), trained across a range of trajectory lengths simultaneously to test generalisation beyond the training distribution

KL divergence between the learned policy and the exact solution is tracked throughout training as the primary evaluation metric, computed efficiently via dynamic programming rather than by enumerating all `2^T` trajectories.

## Key Design Choices

- **KL-regularised reward**: the reward at each step subtracts `log(π(a|s) / 0.5)`, penalising deviation from the uniform reference policy. This keeps the learned sampler stochastic and prevents entropy collapse.
- **Generalisation across T**: the neural policy is conditioned on `(x, t, T)` and trained on trajectory lengths `T ∈ [10, 100]`, then evaluated on unseen lengths up to `T = 200`.
- **Async KL evaluation**: KL computation during neural training is dispatched asynchronously via `@spawn` to avoid blocking the training loop.
- **HPC support**: SLURM batch scripts are included for running multiple random seeds in parallel.

## Stack

- **Julia** with [Lux.jl](https://lux.csail.mit.edu/) (neural networks), [Enzyme.jl](https://enzyme.mit.edu/) (autodiff), [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) (Adam), [CairoMakie.jl](https://docs.makie.org/stable/) (plotting)

## Structure

```
├── src/
│   ├── types.jl          # Core structs (ExcursionProblem, Trajectory, ExactSolution, ...)
│   ├── problem_setup.jl  # Reward, transitions, return computation
│   ├── tabular.jl        # State space iterator, policy iteration, tabular REINFORCE
│   ├── neuralnet.jl      # Neural REINFORCE, actor-critic, KL evaluation
│   └── plotters.jl       # All plotting functions (CairoMakie)
├── data/                 # Saved exact solutions (.jld2) and training runs (.csv)
├── plots/                # Output figures
├── hpc/                  # SLURM scheduler scripts
├── run.jl                # CLI entrypoint
└── Project.toml
```

## Usage

```julia
# Compute and save exact solutions
julia run.jl exact_sols

# Train (single run)
julia run.jl train <seed> <T_step> <T_max> <T_min>

# Plot results
julia run.jl plot_returns
julia run.jl plot_kl
julia run.jl plot_all
```

For multi-seed runs on a SLURM cluster, see `hpc/batch_sheduler.sh`.

## Results

The learned neural policy generalises well to trajectory lengths outside the training range, recovering KL divergence trends consistent with the exact solution. KL regularisation is shown to prevent entropy collapse across all methods. Results are averaged over 10 independent seeds.

## Context

This project is part of ongoing PhD research at the University of Nottingham (School of Physics and Astronomy) into rare event analysis in stochastic systems using reinforcement learning.
