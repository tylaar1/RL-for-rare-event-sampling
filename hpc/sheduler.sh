#!/bin/bash

#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=10:00:00
#SBATCH --output=outs/shortq_%j.out

FUNC="train" #choose from "train", "exact_sols", "plot_returns", "plot_kl", "plot_all"

RANDOM_SEED=0 #allows use of args in this code rather than hardcoding them in the script
T_STEP=2
T_MAX=100
T_MIN=100
NORMIALISE=0
module purge
module load julia-uoneasy
julia --project=. -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
#julia --project=. -e 'using Pkg; Pkg.add("Plots")'
julia --project=. run.jl $FUNC $RANDOM_SEED $T_STEP $T_MAX $T_MIN $NORMIALISE