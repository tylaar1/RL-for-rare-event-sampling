#!/bin/bash

#SBATCH --partition=shortq
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=0:30:00
#SBATCH --output=outs/shortq_%j.out

module purge
module load julia-uoneasy
julia --project=. -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
#julia --project=. -e 'using Pkg; Pkg.add("JLD2")'
julia --project=. run.jl