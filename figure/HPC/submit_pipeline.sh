#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

cd "${SCRIPT_DIR}"

ARRAY_SPEC="$("${PYTHON_BIN}" "${SCRIPT_DIR}/data.py" array-spec "$@")"
ARRAY_JOB_ID="$("${SBATCH_BIN}" --parsable --export="ALL,HPC_DIR=${SCRIPT_DIR}" --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_array.slurm" "$@")"
COMBINE_JOB_ID="$("${SBATCH_BIN}" --parsable --export="ALL,HPC_DIR=${SCRIPT_DIR}" --dependency="afterok:${ARRAY_JOB_ID}" "${SCRIPT_DIR}/combine.slurm" "$@")"
PLOT_JOB_ID="$("${SBATCH_BIN}" --parsable --export="ALL,HPC_DIR=${SCRIPT_DIR}" --dependency="afterok:${COMBINE_JOB_ID}" "${SCRIPT_DIR}/plot.slurm")"

echo "array job:   ${ARRAY_JOB_ID}"
echo "combine job: ${COMBINE_JOB_ID}"
echo "plot job:    ${PLOT_JOB_ID}"
