# Gaia2 Runner

The runner is the host-side entrypoint for Gaia2 evals. It launches containers,
passes agent and judge configuration, collects artifacts, and regenerates the
trace viewer while runs are still in progress.

If you only want to use this repository as a benchmark harness, this is the
piece you will interact with most.

If you need Gaia2 context first, see the
[`Gaia2 evaluation guide`](../../docs/user_guide/gaia2_evaluation.rst) for the
benchmark structure and the
[`scenario foundations`](../../docs/foundations/scenarios.rst) for how
scenarios are built and validated.

## Architecture

```text
TOML config / scenario path / dataset source
                  |
                  v
           gaia2-runner CLI
   - resolves target scenarios
   - loads .env if present
   - chooses image / provider / judge config
   - launches containers with podman
   - mirrors optional host proxy / CA settings
   - writes run_config.json and results.jsonl
                  |
                  v
         per-scenario artifact dirs
   - result.json
   - daemon_status.json
   - events.jsonl
   - trace.jsonl
   - trace.html
                  |
                  v
      view / serve regenerate index.html
```

The runner treats each container as the same abstract backend: send the initial
task over `/notify`, poll `/messages` or `/status`, and then collect artifacts
after completion.

## Commands

| Command | Use it for |
|---------|------------|
| `run` | one scenario |
| `run-dataset` | one split, a dataset directory, or a HuggingFace dataset |
| `run-config` | repeatable launches from a TOML config |
| `serve` | live monitoring during a run |
| `view` | regenerating `index.html` after a run |

Recommended invocation:

```bash
uv run --project runner --python 3.12 gaia2-runner --help
```

## Dataset

The benchmark scenarios are hosted on HuggingFace at
[`meta-agents-research-environments/gaia2-cli`](https://huggingface.co/datasets/meta-agents-research-environments/gaia2-cli).
The runner downloads and caches them automatically when you use
`--dataset meta-agents-research-environments/gaia2-cli` or set
`dataset = "meta-agents-research-environments/gaia2-cli"` in a TOML config.
No manual export is required for normal runner usage.

`HF_TOKEN` is optional for the public dataset, but recommended if you want to
avoid anonymous-download throttling or rate limits.

If you still want local scenario JSON files for inspection, use the included
download script:

```bash
python scripts/export_hf_to_json.py --splits adaptability --dest ./my_dataset
export GAIA2_DATASET_DIR="$(pwd)/my_dataset"
```

Available splits: `adaptability`, `ambiguity`, `execution`, `search`, `time`.

## Quick Start

Recommended first path: run a shipped config against the public Hugging Face
dataset.

```bash
uv run --project runner --python 3.12 gaia2-runner run-config \
    --config runner/examples/quickstart_hermes.toml
```

Retry only infrastructure errors from an existing config output directory:

```bash
uv run --project runner --python 3.12 gaia2-runner run-config \
    --config runner/examples/quickstart_hermes.toml \
    --retry
```

`--retry` reruns only errored scenarios such as startup failures or timeouts.
It does not rerun scenarios that completed and received a benchmark failure.

The easiest first configs are `runner/examples/quickstart_hermes.toml` and
`runner/examples/quickstart_openclaw.toml`. They use one Anthropic key for
both agent and judge. The shipped quickstart and example configs default to
Anthropic Sonnet 4.6 as the judge for operational simplicity. Separately, we
calibrated `gpt-oss-120b` with `low` reasoning as the reference judge
configuration.

If you prefer direct CLI flags, run the public Hugging Face dataset directly:

```bash
uv run --project runner --python 3.12 gaia2-runner run-dataset \
    --dataset meta-agents-research-environments/gaia2-cli \
    --splits search \
    --image localhost/gaia2-oc:latest \
    --provider anthropic \
    --model claude-opus-4-6 \
    --judge-provider anthropic \
    --judge-model claude-sonnet-4-6 \
    --concurrency 20 \
    --output-dir /tmp/gaia2_search
```

Run from a local directory:

```bash
export GAIA2_DATASET_DIR="$HOME/gaia2_datasets/gaia2-cli"

uv run --project runner --python 3.12 gaia2-runner run-dataset \
    --dataset "$GAIA2_DATASET_DIR/search" \
    --image localhost/gaia2-oc:latest \
    --provider anthropic \
    --model claude-opus-4-6 \
    --judge-provider anthropic \
    --judge-model claude-opus-4-6 \
    --concurrency 20 \
    --output-dir /tmp/gaia2_search \
    --output /tmp/gaia2_search/results.jsonl
```

Run one local scenario:

```bash
uv run --project runner --python 3.12 gaia2-runner run \
    --scenario /path/to/scenario.json \
    --image localhost/gaia2-oc:latest \
    --provider anthropic \
    --model claude-opus-4-6 \
    --judge-provider anthropic \
    --judge-model claude-sonnet-4-6 \
    --output-dir /tmp/gaia2_smoke
```

## Config Files

The runner config format is intentionally flat. A config has four sections:

```toml
[target]
dataset = "meta-agents-research-environments/gaia2-cli"
splits = ["search", "time"]

[agent]
image = "localhost/gaia2-oc:latest"
provider = "openai-compat"
model = "your-model"
api_key_env = "OPENAI_COMPAT_API_KEY"
base_url = "https://api.example.com/v1"
thinking = "high"

[judge]
provider = "openai-compat"
model = "your-judge-model"
api_key_env = "OPENAI_COMPAT_API_KEY"
base_url = "https://api.example.com/v1"

[run]
timeout = 900
health_timeout = 180
concurrency = 20
pass_at = 3
output_dir = "/tmp/gaia2_eval"
log_level = "INFO"
```

For a practical default judge, the shipped examples use Anthropic Sonnet 4.6.
Separately, we calibrated `gpt-oss-120b` with `low` reasoning as the reference
judge configuration.

Target selection supports:

- `scenario = "/path/to/scenario.json"` for one scenario
- `dataset = "org/name"` to load from HuggingFace (auto-downloads and caches)
- `dataset_root = "..."` to load from a local directory of scenario JSONs
- Combine with `splits = "search"`, `splits = ["search", "time"]`, or `splits = "all"`
- `subset = "/path/to/subset.json"` to limit runs to a manifest

For dataset targets, the runner preserves split subdirectories in the output
tree automatically.

Run settings support:

- `concurrency`
- `pass_at`
- `retry`
- `timeout`
- `health_timeout`

Paths are resolved relative to the config file. Path fields also expand `~`,
`$VAR`, and `${VAR}`.

Curated examples:

- `runner/examples/quickstart_hermes.toml` — README quickstart: Hermes + direct Anthropic, public `search` split, pass@1
- `runner/examples/quickstart_openclaw.toml` — README quickstart: OpenClaw + direct Anthropic, public `search` split, pass@1
- `runner/examples/hermes_opus_gaia2_pass1.toml` — Hermes + direct Anthropic Opus 4.6, public HuggingFace dataset, pass@1
- `runner/examples/hermes_sonnet_gaia2_pass1.toml` — Hermes + direct Anthropic Sonnet 4.6, public HuggingFace dataset, pass@1
- `runner/examples/hermes_google_gaia2_pass1.toml` — Hermes + direct Google AI Studio Gemini 3.1 Pro Preview, public HuggingFace dataset, pass@1
- `runner/examples/hermes_gpt54_gaia2_pass1.toml` — Hermes + direct OpenAI GPT-5.4, public HuggingFace dataset, pass@1
- `runner/examples/openclaw_opus_gaia2_pass1.toml` — OpenClaw + direct Anthropic Opus 4.6, public HuggingFace dataset, pass@1
- `runner/examples/openclaw_sonnet_gaia2_pass1.toml` — OpenClaw + direct Anthropic Sonnet 4.6, public HuggingFace dataset, pass@1
- `runner/examples/openclaw_google_gaia2_pass1.toml` — OpenClaw + direct Google AI Studio Gemini 3.1 Pro Preview, public HuggingFace dataset, pass@1
- `runner/examples/openclaw_gpt54_gaia2_pass1.toml` — OpenClaw + direct OpenAI GPT-5.4, public HuggingFace dataset, pass@1
- `runner/examples/template_hermes_openai_compat.toml` — generic Hermes template for custom OpenAI chat-completions-compatible endpoints
- `runner/examples/template_openclaw_openai_compat.toml` — generic OpenClaw template for custom OpenAI chat-completions-compatible endpoints

## Secrets and `.env`

The runner auto-loads `gaia2-cli/.env` by default, or a custom dotenv file via
`--env-file`.

Use `.env` for:

- provider API keys
- optional `HF_TOKEN` to reduce Hugging Face throttling or rate limits
- optional proxy settings
- optional `GAIA2_PROXY_RELAY_URL`
- optional `GAIA2_CA_BUNDLE`

Do not treat `.env` as benchmark configuration. Judge provider/model/base URL
belong in CLI flags or TOML so runs stay explicit and reproducible.

Recommended config style:

- put `api_key_env = "OPENAI_API_KEY"` in TOML
- put the actual secret in `.env`

The CLI also accepts direct `--api-key` and `--judge-api-key` flags when you
need a one-off override.

## Provider Notes

Common agent-side providers:

- `anthropic`
- `gemini`
- `openai`
- `openai-compat`
- `openai-completions`
- `google`
- `openrouter`

Use:

- `gemini` for Hermes-native Google AI Studio configs
- `openai` for OpenAI Responses-style backends
- `openai-compat` or `openai-completions` for arbitrary
  chat-completions-compatible endpoints
- `google` for OpenClaw-native Google configs

`--base-url` or `base_url = "..."` is forwarded unchanged into the runtime.

Judge settings are always separate from agent settings:

- `--judge-provider`
- `--judge-model`
- optional `--judge-base-url`
- optional `--judge-api-key`

## Monitoring Runs

Serve the viewer while a run is still active:

```bash
uv run --project runner --python 3.12 gaia2-runner serve \
    --output-dir /tmp/gaia2_eval
```

The server watches the output directory, regenerates HTML on a timer, and
serves the latest `index.html`.

For completed runs, regenerate the static viewer without launching the server:

```bash
uv run --project runner --python 3.12 gaia2-runner view \
    --output-dir /tmp/gaia2_eval
```

The trace viewer keeps raw `trace.jsonl` content intact and normalizes provider
formats at render time. Supported trace payloads are documented in
[TRACE_FORMAT.md](TRACE_FORMAT.md).

## Output Layout

Single-scenario or single-pass dataset outputs:

```text
<output-dir>/
├── index.html
├── run_config.json
├── results.jsonl              # dataset runs
└── <scenario_id>/
    ├── agent_response.txt
    ├── daemon_status.json
    ├── events.jsonl
    ├── result.json
    ├── trace.html
    ├── trace.jsonl            # when emitted by the runtime
    └── *.log
```

When `pass_at > 1`, the root contains `run_1/` through `run_N/`. Each run gets
its own `results.jsonl`, `run_config.json`, and `index.html`.

## Networking

The runner can forward normal host proxy settings into containers:

- `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`
- `no_proxy`, `NO_PROXY`

Two additional host-side knobs are supported:

- `GAIA2_PROXY_RELAY_URL` to start a local relay when only the host can use the
  real outbound proxy
- `GAIA2_CA_BUNDLE` to mount a custom CA bundle into runtimes that need it

That logic lives in the runner, not the containers themselves. If you use
`podman run` directly, you need to pass equivalent env vars by hand.
