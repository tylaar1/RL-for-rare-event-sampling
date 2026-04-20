#!/bin/bash

#SBATCH --partition=defq
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=10:00:00

#SBATCH -o ./logs/output-%A_%a.out # STDOUT

TASK=${SLURM_ARRAY_TASK_ID}
"""for kl not training"""
T_STEP=2
T_MAX=200
T_MIN=10
echo "Running Job $TASK on `hostname`"
cd ${SLURM_SUBMIT_DIR}

module load julia-uoneasy/1.10.4-linux-x86_64

julia --project=. -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
julia --project=. run.jl $TASK $T_STEP $T_MAX $T_MIN