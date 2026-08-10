# Sharded OmniGAIA-MT on a SLURM cluster

Run the benchmark across **K parallel SLURM jobs** sharing one judge endpoint
(and optionally one agent endpoint). Compresses pass@3 × all-splits from
~24–42 h on a single node to roughly `40h/K + endpoint startup`.

This supersedes [`RUNBOOK.md`](RUNBOOK.md) §6 for any run large enough to want
more than one node's throughput. Setup from `RUNBOOK.md` §0–§4 (venvs, images,
`.env`) still applies — only the launch and the agent/judge topology change.

> **Container runtime.** Sharded runs on a CPU partition **must** use
> `runtime = "apptainer"` — see [`RUNBOOK.md`](RUNBOOK.md) §"Container runtime:
> podman vs apptainer" for why, and §1b for the one-shot `.sif` build.

## Serving the shared endpoints

The fan-out needs one or two long-lived OpenAI-compatible endpoints reachable
from the shard nodes. How you get them is up to your cluster — an `sbatch` job
running `vllm serve`, a Ray Serve deployment, a managed inference service. We
used a Ray + vLLM serving wrapper with a small endpoint registry in front of
it; nothing downstream depends on that choice.

All the fan-out needs is the resulting URLs:

```
http://<judge-host>:8001/v1
http://<agent-host>:8000/v1
```

Two properties worth reproducing whatever you use:

- **Idempotent start.** Launching twice should reuse a healthy endpoint
  rather than allocating a second copy of the model.
- **A way to query the current URL.** If the scheduler places the server on
  whatever node is free, the hostname is *not* stable across restarts. Always
  re-read the current URL before submitting a fan-out; never copy one from a
  doc or from old shell history.

`/v1/models` answering 200 is **not** sufficient readiness — a Ray Serve or
similar front end answers it before the vLLM replicas finish loading weights.
Drive a real `/chat/completions` ping (see §1) before submitting any shards.

## Runner support for sharding

Three pieces of plumbing, usable independently:

* **`gaia2-runner --shard-id N --num-shards K`** — round-robin partition of
  the scenario list on sorted scenario IDs, applied AFTER `subset` and
  `limit`. Multi-split runs (`splits = ["search","execution"]`) shard the
  CONCATENATED list, so each shard covers all splits proportionally. The
  runner auto-namespaces `output_dir` with `shard_{id:02d}_of_{n:02d}` so
  concurrent shards don't trip the "refuses to overwrite" check. Also
  settable in TOML as `[run].shard_id` / `[run].num_shards`.

* **`gaia2-runner aggregate`** — `gaia2-runner aggregate --output-dir <root>`
  walks `<root>/shard_*_of_*/results.jsonl` and merges into
  `<root>/results.jsonl`. Refuses overlapping `scenario_id`s. Does not
  regenerate `index.html` — run `gaia2-runner serve --output-dir <root>` for
  that.

* **`no_proxy` auto-extension** — the runner extends the in-container
  `no_proxy` with the hosts of `[agent].base_url` and `[judge].base_url`, so
  a remote judge on another node isn't tunneled through OpenClaw's TLS MITM
  proxy.

## Topology

```
                          ┌─────────────────────┐
                          │  Judge: gpt-oss-120b│
                          │  shared endpoint    │
                          │  http://nodeJ:8001  │
                          └──────────┬──────────┘
                                     │ shared across all shards
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────┴───────┐            ┌───────┴───────┐            ┌───────┴───────┐
│ Shard 0/K     │            │ Shard 1/K     │     ...    │ Shard K-1/K   │
│ Agent (TP=2)  │            │ Agent (TP=2)  │            │ Agent (TP=2)  │
│ gaia2-runner  │            │ gaia2-runner  │            │ gaia2-runner  │
│ → shard_00/   │            │ → shard_01/   │            │ → shard_K-1/  │
└───────────────┘            └───────────────┘            └───────────────┘
                                     │
                                     ▼
                          gaia2-runner aggregate
                                     │
                                     ▼
                          <out_root>/results.jsonl
```

GPU budget: `2 × (K+1)` GPUs if both agent and judge are TP=2. The judge can
be reused across many sharded runs over days, so the +1 is amortized.

**Cheaper still:** when comparing one agent model across several languages,
bring the agent up **once** as a shared endpoint and point every shard at it.
Each shard becomes a CPU-only HTTP client (`--gpus-per-task=0`) and the GPU
budget collapses to `agent_tp + judge_tp` total, independent of K and of the
number of languages. This is the topology the example TOMLs in this directory
assume.

## 1. Bring up the shared endpoints (once)

Whatever tool you use, the result should be reachable from the shard nodes by
hostname. Verify before submitting anything:

```bash
curl -sS http://<agent-host>:8000/v1/models | jq .
curl -sS http://<judge-host>:8001/v1/models | jq .
```

Then confirm the model actually generates, not just that the route exists:

```bash
curl -sS http://<judge-host>:8001/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
```

On clusters with an egress proxy, clear it before talking to in-cluster hosts
(`HTTP_PROXY= HTTPS_PROXY= …`) or the request will be tunneled out and back.

## 2. Translate the dataset (one-time per language)

Unchanged from [`RUNBOOK.md`](RUNBOOK.md) §7a. Once translated, the scenario
JSON lives under `$OUTPUT_BASE/<lang>/data/`.

## 3. Fan out shards

Start from [`omnigaia_mt_apptainer_sharded.toml`](omnigaia_mt_apptainer_sharded.toml)
— all four splits, `runtime = "apptainer"`, `idle_timeout = 600.0`. Edit
`dataset_root`, `output_dir` and the two `base_url` values for the endpoints
from §1. Two shard-specific notes:

- Leave `shard_id` / `num_shards` out of the base TOML — pass them per job.
  `output_dir` is auto-namespaced with `shard_{id:02d}_of_{n:02d}`, so a
  hard-coded per-shard subdir would make the shards collide.
- `[agent].model` must match the `model_id` your server advertises
  **exactly**, or every scenario fails with a 404 from the agent endpoint.

Then submit one job per shard. Nothing about the fan-out is gaia2-specific —
any scheduler loop that sets `--shard-id`/`--num-shards` works:

```bash
for i in $(seq 0 7); do
  sbatch --cpus-per-task=32 --mem=64G --wrap \
    "gaia2-runner run-config --config base.toml --shard-id $i --num-shards 8"
done
gaia2-runner aggregate --output-dir <out>
```

Sizing notes from our runs:

- **One shard per node.** The in-container OpenClaw stack (TLS proxy + gateway
  + adapter + eventd, all racing at startup) does not tolerate co-located
  shards. Packing 8 shards onto one node produced ~85% ERROR. Add
  `--exclusive` (or equivalent); CPU nodes are usually plentiful.
- **`--cpus-per-task=32` is sized for `concurrency = 16`.** Each scenario
  starts an adapter, a daemon and a gateway, all Python. Under-provisioning
  CPU starves the adapter health probes and trips the 180 s timeout.
- **`--gpus-per-task=0`** when the agent is a shared endpoint; `2` if each
  shard brings up its own TP=2 agent.
- Export `GAIA2_OC_SIF` into the job environment (or set `[agent].image_sif`)
  when using apptainer, and `GAIA2_OC_SIF_STAGE_LOCAL=1` to stage the `.sif`
  to node-local `/scratch` — see "Apptainer-specific notes" below.

### Shared agent (multi-language sweeps)

Bring the agent up once and loop the languages, each with its own base TOML
differing only in `dataset_root` and `output_dir`:

```bash
for LANG in eng spa fra deu zho; do
  for i in $(seq 0 3); do
    sbatch --cpus-per-task=32 --mem=64G --wrap \
      "gaia2-runner run-config --config configs/omnigaia_mt_${LANG}.toml \
         --shard-id $i --num-shards 4"
  done
done
```

Each shard writes to `<output_dir>/shard_00_of_04/`, `shard_01_of_04/`, …
Per-shard `results.jsonl`, `index.html` and traces behave identically — the
shard just sees a smaller scenario list.

## 4. Aggregate

After all shards finish (watch with `squeue -u $USER`):

```bash
gaia2-runner aggregate --output-dir /path/to/scratch/$USER/omnigaia/eval/spa_full
# aggregated 800 results from 4/4 shards
```

`<output_dir>/results.jsonl` now holds the full benchmark. For an interactive
viewer, `gaia2-runner serve --output-dir <output_dir>`.

## Apptainer-specific notes

### `.sif` staging to node-local `/scratch`

Set `GAIA2_OC_SIF_STAGE_LOCAL=1` in the shard job's environment.
`ApptainerLauncher` reads that variable and copies the `.sif` from
shared storage to `/scratch/$USER/apptainer/<basename>` once per node before
launching any instances. Subsequent instances on that node mount FUSE from
local disk and share a warm page cache.

Without it, 8×4 = 32 concurrent FUSE mounts of one ~500 MB NFS file were
enough to peg the adapter health probes — every scenario in the first wave hit
the 180 s timeout and hard-ERRORed. Staging is a ~5 s one-time-per-node copy.
The flag is **off by default** in the launcher so workstations without
`/scratch` are unaffected.

### Why `--writable-tmpfs`, not `--overlay <dir>`

The launcher needs a per-instance writable rootfs layer so the daemon can
write `/tmp/gaia2-{adapter,eventd,openclaw}.log`. `--overlay <host-dir>` is
the tempting choice — host-backed, persistent for post-mortem, no RAM cost —
and it works on most nodes. It does **not** work on nodes whose host rootfs is
itself containerd-overlayfs (`stat -f /tmp` → `Type: overlayfs`). Nesting
apptainer's overlay on top strips exec permissions on `/usr/bin/bash` for
non-root uids under `--fakeroot`, so the entrypoint's
`su -s /usr/bin/bash gaia2 -c …` fails with EACCES and the adapter never
binds. Symptom: 100% of the shard's scenarios report
`Adapter on port X not ready after 180s`.

`--writable-tmpfs` sidesteps the nested-overlay path entirely, and on
apptainer 1.5.1 it is genuinely per-instance. RAM cost is small because the
daemon's logs are small.

For forensics, `GAIA2_OC_KEEP_SCRATCH=1` makes `ApptainerLauncher.stop()`
preserve the per-instance bind dirs
(`/scratch/$USER/apptainer/<instance>/{state,tmp}`) so you can read
`entrypoint.out` and `tmp/gaia2-*.log` after the fact.

## Apptainer TOML knobs

```toml
[agent]
runtime   = "apptainer"            # vs "podman"; dispatches the launcher
image_sif = ".../gaia2-oc.sif"     # absolute path; overridable via $GAIA2_OC_SIF

[run]
concurrency  = 4       # apptainer instances in flight per shard
idle_timeout = 600.0   # seconds w/o new daemon events before giving up
timeout      = 1200    # hard wall-clock per scenario
stuck_loop_min_tool_calls = 3   # daemon status=error + this many LLM calls
                                # → reclassified FAIL (model loop), not ERROR.
                                # Set to 0 to disable.
```

`stuck_loop_min_tool_calls` keeps the ERROR bucket pure-infra: a daemon that
ends in `status=error` after the agent already made N+ LLM calls without a
turn boundary is a runaway tool-loop — a real model FAIL, not an adapter
hiccup. Tune it up if you want short loops counted as ERROR.

## Scaling up gradually

Each rung costs roughly 5× the previous and catches a different class of bug.
Don't skip ahead.

| Rung | Scenarios | Shards | Concurrency | Catches |
|---|---|---|---|---|
| smoke | 1 | 1 | 1 | `.sif` boots, adapter binds, scenario completes |
| midsmoke | 32 | 2 | 4 | Instance churn, parallel adapter startup, port allocation, judge throughput |
| full | 640–800 | 8–16 | 4–8 | Tail behaviours, runaway scenarios, day-long endpoint stability |

Judge the rungs on **ERROR count**, not PASS rate: PASS/FAIL is a
model-quality measurement, ERROR is the infra signal. More than ~1–2 ERROR out
of 32 at the midsmoke rung means something regressed — investigate before
scaling. Pass-rate skew *between* shards is expected and not a bug:
round-robin sharding on sorted scenario paths can hand one shard a harder
slice, and at n≈16/shard the variance is high.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| All scenarios FAIL with 404 from the agent | `[agent].model` doesn't match the `model_id` the server advertises. |
| Shards stuck at `[shard] preflight: waiting for http://…/v1 to serve model=…` for >5 min | The endpoint URL is stale — the server moved to a different node on its last restart. Re-read the current URL and resubmit. |
| Shard hangs in "waiting for agent to serve", then aborts | The endpoint answered `/v1/models` but the vLLM replicas hadn't loaded. Wait, or restart the server. |
| All shards spin up but only shard 0 writes output | Check `output_dir` namespacing. The runner appends `shard_{id:02d}_of_{n:02d}` automatically when `num_shards > 1`; a hard-coded per-shard subdir in the TOML makes them collide. |
| `gaia2-runner aggregate` raises `overlapping scenario_id` | Two shards processed the same scenario — usually different `num_shards` values across the per-shard configs, or `--shard-id` set without `--num-shards`. |
| Judge OOM mid-run | The judge sees K × concurrency simultaneous requests. Raise its `tensor_parallel_size` or lower the per-shard `concurrency`. |
| `runtime=apptainer in <toml> but neither [agent].image_sif nor $GAIA2_OC_SIF is set` | Set `[agent].image_sif` in the base TOML, or export `GAIA2_OC_SIF` into each shard job's environment. |
| 100% of a shard's scenarios ERROR with `Adapter on port X not ready after 180s`, and `entrypoint.out` shows `su: failed to execute /usr/bin/bash: Permission denied` | Nested-overlay EACCES. The launcher must use `--writable-tmpfs`, not `--overlay <dir>` — see above. |
| Every scenario in the first wave times out at exactly 180 s on adapter health | NFS-hosted `.sif` thrashing under parallel FUSE mounts. Confirm the SLURM log shows `Staging .sif from … to /scratch/…`. If staging is happening and it still times out, check node-local `/scratch` space (`df -h /scratch`). |
| ~30% ERROR, several shards landed on the same node | `SBATCH_NO_EXCLUSIVE=1` was set. Drop it; the OpenClaw startup race needs one shard per node. |
| Shards fail with `chown: Operation not permitted` during image extraction | The base config is still `runtime = "podman"` on an overlayfs `/scratch`. Switch to apptainer and provide `image_sif`. |
| `apptainer instance start` fails with overlay errors | `.sif` is corrupt or was built on an incompatible host; rebuild it. |
| Scenario hits the 1200 s wall clock with a huge daemon event count | Runaway tool-loop — a real model error, not infra. |
| Live progress shows P+F but the `F=` column looks inflated | The tqdm `F=` field aggregates FAIL+ERROR. Read the `daemon_status` field of `results.jsonl` for the real split. |

## Reference

* [`RUNBOOK.md`](RUNBOOK.md) — single-node quickstart.
* [`gaia2-cli/runner/README.md`](../../README.md) — runner reference, output layout, trace format.
* [`gaia2-cli/mt/README.md`](../../../mt/README.md) — OmniGAIA-MT translation pipeline.
