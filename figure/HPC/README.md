# HPC Workflow

This folder contains a SLURM job-array version of the random-regular two-replica sweep.

Default production setting: `T=600000`.

## Files

- `data.py`: computes one `mu` point per task and combines all shard files into a single `output/data.npz`.
- `plot.py`: reads `output/data.npz` and writes `output/REGULAR_AT.pdf`.
- `run_array.slurm`: SLURM array worker for one `mu` index.
- `combine.slurm`: SLURM job that combines all shard files.
- `plot.slurm`: SLURM job that renders the final PDF.
- `submit_pipeline.sh`: submits the full chain with dependencies.

## Default Outputs

- `output/shards/mu_XXXX.npz`
- `output/data.npz`
- `output/REGULAR_AT.pdf`

The combined `output/data.npz` preserves the same high-level arrays as the serial workflow and includes `metadata_json`.

## Submit The Full Pipeline

From inside this folder:

```bash
bash submit_pipeline.sh
```

The submit script exports `HPC_DIR` into each SLURM job so the batch node runs the Python files from the real folder rather than SLURM's temporary spool copy.

This will:

1. Submit an array job over all default `mu` values.
2. Submit a combine job that waits for the array to finish successfully.
3. Submit a plot job that waits for the combine step.

## Useful Overrides

Every script forwards the same CLI options to `figure/HPC/data.py`, for example:

```bash
bash submit_pipeline.sh --n-mu-points 80 --g 401 --m 2500 --t 30000 --burn-in 60000
```

Quick smoke test:

```bash
bash submit_pipeline.sh --quick
```

## Manual Submission

If you want to submit the stages yourself:

```bash
python3 data.py array-spec
sbatch --array="$(python3 data.py array-spec)" run_array.slurm
sbatch combine.slurm
sbatch plot.slurm
```

For the manual route, add `--dependency=afterok:<jobid>` yourself between stages.

## Cluster Notes

- Set `PYTHON_BIN` if your cluster uses a non-default Python executable.
- Add partition, account, or qos flags either directly in the `.slurm` files or on the `sbatch` command line.
- The array size is computed from `n_mu_points`, so it stays consistent with the Python configuration.
