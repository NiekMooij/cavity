#!/bin/bash
#SBATCH --job-name=rr_replica_array
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

HPC_DIR="${HPC_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${HPC_DIR}" ]]; then
    echo "HPC_DIR is not set" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${HPC_DIR}"
"${PYTHON_BIN}" "${HPC_DIR}/data.py" shard "$@"
