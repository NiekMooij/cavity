#!/bin/bash
#SBATCH --job-name=rr_replica_plot
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

HPC_DIR="${HPC_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${HPC_DIR}" ]]; then
    echo "HPC_DIR is not set" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${HPC_DIR}"
"${PYTHON_BIN}" "${HPC_DIR}/plot.py"
