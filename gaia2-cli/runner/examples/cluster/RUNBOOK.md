# Gaia2 on a SLURM cluster — single-node runbook

End-to-end recipe for running the Gaia2 benchmark on one GPU node of a SLURM
cluster, plus the OmniGAIA-MT multilingual extension. For the multi-node
fan-out, see [`RUNBOOK.sharded.md`](RUNBOOK.sharded.md).

These are the configs and steps we actually used on the SLURM cluster this was
developed on, generalised where they were site-specific. Treat them as a worked
example rather than a drop-in script: partition names, storage roots and the
model-serving tool will differ on your cluster.

For background on Gaia2 (scenarios, scoring, the broader ARE platform) see the
[top-level README](../../../../README.md) and the
[Gaia2 evaluation guide](../../../../docs/user_guide/gaia2_evaluation.rst).
The generic, off-cluster quickstart is
[`gaia2-cli/README.md`](../../../README.md).

## Reference configuration

| Role  | Model                    | Backend         | GPUs | Port | Served as       |
|-------|--------------------------|-----------------|------|------|-----------------|
| Agent | `google/gemma-4-31B-it`  | local vLLM TP=2 | 0,1  | 8000 | `gemma-4-31b`   |
| Judge | `openai/gpt-oss-120b`    | local vLLM TP=2 | 2,3  | 8001 | `gpt-oss-120b`  |
| Runtime | `localhost/gaia2-oc:latest` (OpenClaw) | podman OR apptainer | — | — | — |

One node, 8× H200.

### Container runtime: podman vs apptainer

The runner supports two container backends, selected per run via
`[agent].runtime` in the TOML:

| Backend | When to use | Where it works |
|---|---|---|
| `podman` (default) | Single-node runs where the node's `/scratch` is a normal filesystem (xfs/ext4). | GPU nodes, devservers. |
| `apptainer` | Any node whose `/scratch` is overlayfs, and sharded fan-out (see [`RUNBOOK.sharded.md`](RUNBOOK.sharded.md)). | Everywhere. The portable choice. |

Podman fails on nodes with an overlayfs `/scratch`: the `vfs` storage driver
needs `CAP_CHOWN` on the upper layer during image extraction, rootless
processes don't have it, and `ignore_chown_errors` doesn't help (the kernel
refuses before vfs gets to ignore anything). Apptainer mounts the image as
read-only SquashFS via FUSE — no chown at runtime, no daemon. Both backends
produce byte-identical scenario artifacts.

Check which you're on with `stat -f /scratch`.

## 0. Prerequisites

On the login node:

- This repo cloned somewhere on shared storage. Run all commands from the
  repo root.
- `podman`, `tmux`, `uv`, `python 3.12` available.
- Hugging Face cache pointed at a shared mirror so you don't re-download
  models and datasets per user:
  ```bash
  export HF_HOME=/path/to/shared/huggingface
  export HF_HUB_CACHE=$HF_HOME/hub
  ```
  Add these to your shell rc to make them sticky. If something is missing
  from the cache, `huggingface-cli login` once and re-run; the download
  lands in the shared cache for the next person.

## 1. Get the container images onto the cluster

If your login node can build images, the upstream `make gaia2-oc` is all you
need. Many can't — on ours, `podman build` failed at the first `RUN` with
`mount /proc: Operation not permitted` (cgroups v1 plus a restrictive
seccomp profile). In that case build elsewhere and ship an archive:

```bash
# On a machine that can build:
make gaia2-oc
podman save --format oci-archive -o gaia2-images.tar \
    localhost/gaia2-oc:latest localhost/gaia2-cli:local
```

Use `oci-archive`, not `docker-archive` — the latter does not support
multiple images per file. Copy the tar to shared storage, then on the
cluster:

```bash
export GAIA2_IMAGES_TAR=/path/to/shared/images/gaia2-images.tar
podman load -i "$GAIA2_IMAGES_TAR"
podman images | grep -E 'gaia2-(oc|cli)'
# expect: localhost/gaia2-oc:latest  and  localhost/gaia2-cli:local
```

### 1b. Build the apptainer `.sif` (only if you'll use `runtime = "apptainer"`)

Apptainer consumes a SquashFS `.sif`, not an OCI tag. Convert the archive
once, on a compute node that has apptainer plus `fakeroot` subuid mappings
(login nodes typically lack both):

```bash
srun --partition=<cpu-partition> --time=15:00 --cpus-per-task=4 --mem=16G \
     bash gaia2-cli/scripts/build_apptainer_image.sh
ls -lh gaia2-cli/gaia2-images/gaia2-oc.sif
```

The script reads `GAIA2_OC_ARCHIVE` (default `gaia2-cli/gaia2-images.tar`)
and writes `GAIA2_OC_SIF` (default `gaia2-cli/gaia2-images/gaia2-oc.sif`).
The `.sif` is portable and read-only; rebuild only when the OCI archive
changes. Takes ~2 min and writes ~500 MB.

To override the archive path under `srun`, use SLURM's own env propagation
(`--export=ALL,KEY=VAL`) rather than `env KEY=VAL bash …` — the latter makes
SLURM try to exec `$HOME/.local/bin/env` and trip a permission error:

```bash
srun --partition=<cpu-partition> --time=20:00 --cpus-per-task=4 --mem=16G \
     --export=ALL,GAIA2_OC_ARCHIVE=/path/to/shared/images/gaia2-images.tar \
     bash gaia2-cli/scripts/build_apptainer_image.sh
```

Alternatively, if you publish the image to a container registry, apptainer
can convert directly and you can skip the tar entirely:

```bash
apptainer build gaia2-oc.sif docker://<registry>/gaia2-oc:<version>
```

Do **not** hand-write an equivalent `.def` recipe: a `--fakeroot` apptainer
build silently strips the setuid bit on `gaia2-exec`, which is the
agent→`gaia2` privilege boundary the sandbox model depends on. The OCI→
SquashFS conversion preserves it.

## 2. Create two venvs (one-time)

vLLM's torch/CUDA wheels conflict with the runner's deps. Keep them split:

```bash
# Runner: CLI + orchestrator
uv venv -p 3.12 ~/.venvs/gaia2-runner
source ~/.venvs/gaia2-runner/bin/activate
uv pip install -e gaia2-cli/cli -e gaia2-cli/runner
deactivate
```

```bash
# vLLM: just vllm
uv venv -p 3.12 ~/.venvs/vllm
source ~/.venvs/vllm/bin/activate
uv pip install "vllm>=0.10.2"
deactivate
```

## 3. Drop a `.env` for the runner

```bash
cat > gaia2-cli/.env <<'EOF'
OPENAI_COMPAT_API_KEY=dummy
HF_TOKEN=hf_xxx
no_proxy=localhost,127.0.0.1,::1
NO_PROXY=localhost,127.0.0.1,::1
EOF
```

The `no_proxy` lines are required: OpenClaw runs its own in-container TLS
MITM proxy and would otherwise tunnel host-loopback vLLM traffic through it.

## 4. Start both vLLM servers

Two servers, one per role, on separate GPUs. Run each in its own terminal (or
`tmux` window) from the vLLM venv:

```bash
source ~/.venvs/vllm/bin/activate

# Agent, GPUs 0-1, port 8000
CUDA_VISIBLE_DEVICES=0,1 vllm serve google/gemma-4-31B-it \
    --served-model-name gemma-4-31b --port 8000 \
    --tensor-parallel-size 2 --max-model-len 131072 \
    --tool-call-parser gemma4 --enable-auto-tool-choice

# Judge, GPUs 2-3, port 8001
CUDA_VISIBLE_DEVICES=2,3 vllm serve openai/gpt-oss-120b \
    --served-model-name gpt-oss-120b --port 8001 \
    --tensor-parallel-size 2 --max-model-len 131072
```

`--served-model-name` must match `[agent].model` / `[judge].model` in the TOML
exactly, or every request 404s. Wait until each server answers a real
completion — `/v1/models` returning 200 is not sufficient readiness:

```bash
curl -sS http://localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"gemma-4-31b","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
```

Two flags are load-bearing and model-specific:

- `--tool-call-parser` — use the correct parser for your model (`gemma4` for
  Gemma 4). With the wrong one, every scenario scores 0 and `agent_response`
  in `result.json` is raw unparsed `call:exec{...}` text.
- `--max-model-len 131072` — Gaia2 prompts are long. Below ~33k the agent
  starts hitting "maximum context length exceeded".

## 5. Run a subset (recommended first run)

A working config is at
[`local_vllm_oc_gemma.toml`](local_vllm_oc_gemma.toml). Copy and edit for a
smoke run:

```toml
[target]
splits = ["search"]   # one capability split (of 5: search, execution,
                      # adaptability, time, ambiguity). Use "all" for the
                      # full 800-scenario benchmark.
limit  = 10

[run]
concurrency = 4
pass_at     = 3
output_dir  = "/path/to/scratch/$USER/omnigaia/<run_name>"
```

`pass_at > 1` requires `output_dir`. Use a **fresh** `output_dir` per run —
the runner refuses to overwrite an existing one. Put it on persistent shared
storage, not node-local scratch.

To run with apptainer instead of podman, add to `[agent]`:

```toml
[agent]
runtime   = "apptainer"
image_sif = "/path/to/gaia2-images/gaia2-oc.sif"
# image is still required (apptainer reuses it as a label only)
image     = "localhost/gaia2-oc:latest"
```

The `.sif` path can also come from `$GAIA2_OC_SIF` (env var wins over the
TOML value). All other config keys are identical between backends.

Launch:

```bash
source ~/.venvs/gaia2-runner/bin/activate
gaia2-runner run-config --config gaia2-cli/runner/examples/cluster/local_vllm_oc_gemma.toml
```

When it finishes, open `<output_dir>/index.html`. Raw per-scenario traces
live at `<output_dir>/run_{1,2,3}/<split>/<scenario_id>/result.json`.

Reference timing: pass@3 / `splits=["search"]` / `limit=10` takes ~15 min on
the configuration above.

### Inspecting results

Each `<scenario_id>/` directory contains:

- `result.json` — pass/fail, scores, model + judge metadata. Start here.
- `trace.html` — rendered transcript with tool calls; easiest entry point
  for failure analysis.
- `trace.jsonl` — raw per-turn LLM API records (request, response, latency).
  Use for programmatic analysis.
- `events.jsonl` — scenario-level events (oracle checkpoints, env state
  changes).
- `agent_response.txt` — the agent's final reply.
- `daemon_status.json`, `*.log` — runtime + container logs for debugging
  infra failures.

The aggregate `<output_dir>/index.html` links to every scenario's
`trace.html`. For runs in progress, or to regenerate `index.html` after the
fact, serve the directory:

```bash
gaia2-runner serve --output-dir <output_dir>
```

Full output layout — including the `run_N/` split under `pass_at > 1` — is
documented in
[`gaia2-cli/runner/README.md`](../../README.md#output-layout). The
`trace.jsonl` schema is in
[`gaia2-cli/runner/TRACE_FORMAT.md`](../../TRACE_FORMAT.md).

## 6. Run the full benchmark

Same config as step 5 but drop `limit` and set `splits = "all"` (5 splits ×
160 = 800 scenarios). Wall clock with TP=2 and `concurrency=8` is **8–14 h
for pass@1**; pass@3 is ×3. If that's too slow, shard it — see
[`RUNBOOK.sharded.md`](RUNBOOK.sharded.md).

## 7. OmniGAIA-MT: translate + evaluate multilingual

OmniGAIA-MT extends the run above to non-English benchmarks: it translates
the Gaia2 scenarios into a target language, then evaluates with the OmniGAIA
multilingual judge prompts. Both halves can use the same local vLLM servers
from steps 2–4.

The translation package lives in [`gaia2-cli/mt/`](../../../mt/README.md);
the judge prompt overrides live in the gaia2-cli core judge
(`gaia2_core.judge.prompt_overrides`, version `omnigaia`) and are selected
per run via `[judge].prompt_version`.

### 7a. Install and translate

Follow [`gaia2-cli/mt/README.md`](../../../mt/README.md) — a third venv, then
`run_translate.sh` (or `python -m omnigaia.cli.translate`) pointed at the agent
vLLM server from step 4 via `OMNIGAIA_LLM_BASE_URL`.

The handoff back to this runbook: the translator's `output_dir` — one
subdirectory per split — is consumed verbatim as `[target].dataset_root`
below, so write it to shared storage the eval nodes can read.

### 7b. Evaluate with the OmniGAIA judge

Use the ready-made config [`omnigaia_mt.toml`](omnigaia_mt.toml) — same
agent/judge vLLM setup as `local_vllm_oc_gemma.toml`, but pointed at the
translated `dataset_root` with `[judge].prompt_version = "omnigaia"`:

```bash
source ~/.venvs/gaia2-runner/bin/activate
gaia2-runner run-config --config gaia2-cli/runner/examples/cluster/omnigaia_mt.toml
```

The runner prints `Judge prompts: omnigaia` in its summary when the
override is active. Set `prompt_version = "default"` to score the same
translated data against the stock gaia2-core prompts for an apples-to-apples
judge comparison.

Scenario data stays **outside** the container: the runner bind-mounts each
scenario JSON per-instance from `[target].dataset_root` to
`/var/gaia2/custom_scenario.json:ro`. Note this is distinct from
`/opt/gaia2_filesystem` inside the image, which holds the small, stable
filesystem *fixtures* scenarios reference via `real_path`.

## 8. Re-score a finished run offline

Judge verdicts are the expensive, noisy half of a run. When you want to change
*only* the judge — a different judge model, `prompt_version = "default"` vs
`"omnigaia"`, or an `extra_body` tweak — replaying the recorded action
stream is far cheaper than re-running the agent:

```bash
source ~/.venvs/gaia2-runner/bin/activate
python -m gaia2_cli.judge.rejudge \
    --scenario-dir <output_dir>/run_1/<split>/<scenario_id> \
    --out /tmp/replay/replay_judgments.jsonl \
    --judge-model gpt-oss-120b \
    --judge-provider openai-compat \
    --judge-base-url http://localhost:8001/v1 \
    --judge-prompt-version omnigaia
```

It reads `events.jsonl` and `result.json` from the scenario dir, rebuilds the
same `Judge` the daemon used, and re-scores turn by turn. Nothing in the
original directory is modified; verdicts go to `--out`, and a `daemon_diff`
summary compares them to the `daemon_judgments.jsonl` already on disk.

Use it for judge-prompt A/B comparisons and for judge-flakiness triage. Do
**not** use it to check agent-side changes — the agent is never re-invoked, so
the action stream is fixed. Omit `--judge-model` to replay with deterministic
argument matching only (no LLM checkers), which is what the unit tests do.

Point `--out` at a directory of its own: the `Judge` also writes its detailed
`judgments.jsonl` into `--out`'s parent.

## Failure-mode quick reference

| Symptom | Cause / fix |
|---|---|
| `agent_response` in `result.json` is raw `call:exec{...}` text, every scenario scores 0 | Wrong tool-call parser. Confirm `--tool-call-parser` matches your model, restart the agent server. |
| vLLM rejects requests with "maximum context length exceeded" | `--max-model-len` too low. Restart with 131072. |
| `podman build` on the login node fails with `mount /proc: Operation not permitted` | Expected on restricted login nodes. Build elsewhere and load an archive (step 1). |
| Judge errors on every scenario | `gpt-oss-120b` may need `reasoning_effort="low"` — see `gaia2-cli/cli/gaia2_cli/judge/engine.py`. |
| OOM on the agent GPU at boot | Drop `--max-model-len` to 65536, or set `--gpu-memory-utilization 0.85`. |
| First scenario's container won't start | `podman images \| grep gaia2-oc` — if missing, re-run `podman load` (step 1). |
| MT translation can't reach the model, or tries the default hosted endpoint | `OMNIGAIA_LLM_BASE_URL` not set. Export it (plus `OMNIGAIA_LLM_API_KEY`) to point at your vLLM server. |
| MT eval ignores the OmniGAIA judge prompts | Confirm `[judge].prompt_version = "omnigaia"`; the run summary should print `Judge prompts: omnigaia`. Unknown versions log a warning and fall back to defaults. |
| `runtime = "apptainer"` but the runner errors with "neither `image_sif` nor `$GAIA2_OC_SIF` is set" | Set `[agent].image_sif` in the TOML or export `GAIA2_OC_SIF`. See §1b for the build. |
| `podman` errors with `chown: Operation not permitted` during image extraction | The node's `/scratch` is overlayfs (`stat -f /scratch`). Switch to `runtime = "apptainer"` or move the job to a node with a normal `/scratch`. |
