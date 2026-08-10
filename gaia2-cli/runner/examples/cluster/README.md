# SLURM cluster examples

Worked configurations for running Gaia2 and OmniGAIA-MT on a SLURM cluster.
These are the files we actually used, with site-specific paths, hostnames and
scheduling knobs replaced by placeholders. They are **starting points, not drop-in configs** — expect to edit every
`/path/to/...` and every endpoint URL before running.

## Runbooks

| Doc | Scope |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | Single node: images, venvs, local vLLM, one run, then the OmniGAIA-MT translate + evaluate loop. Start here. |
| [`RUNBOOK.sharded.md`](RUNBOOK.sharded.md) | Multi-node fan-out: K SLURM jobs against shared agent/judge endpoints, plus aggregation and the apptainer-specific operational notes. |

## Configs

| TOML | What it is |
|---|---|
| `local_vllm_oc_gemma.toml` | Baseline single-node Gaia2 run against two local vLLM servers (agent + judge). |
| `omnigaia_mt.toml` | Same topology, pointed at a translated `dataset_root` with `[judge].prompt_version = "omnigaia"`. |
| `omnigaia_mt_apptainer_sharded.toml` | All four splits, `runtime = "apptainer"`, raised `idle_timeout` — the base config for a sharded fan-out. Header comments cover the locale / smoke / pass@3 variants. |

Per-shard TOMLs (`<base>.shard_NN_of_KK.toml`) and their run artifacts are
gitignored by the repo-root `.gitignore`.

## Choosing a container runtime

`podman` is the default; `runtime = "apptainer"` is required where `/scratch`
is overlayfs. Full rationale in [`RUNBOOK.md`](RUNBOOK.md) §"Container runtime:
podman vs apptainer"; the `.sif` build is §1b.

## Scripts these examples drive

- [`gaia2-cli/scripts/build_apptainer_image.sh`](../../../scripts/build_apptainer_image.sh)
  — convert the OCI archive to a `.sif`.
