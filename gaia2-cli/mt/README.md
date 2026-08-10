# OmniGAIA-MT: multilingual translation pipeline

OmniGAIA-MT translates the Gaia2 benchmark into a target language, producing a
`dataset_root` directory of translated scenario JSON files that the
[gaia2-cli runner](../runner/README.md) evaluates directly. Paired with the
`omnigaia` judge prompts (ported into `gaia2_core.judge`), it runs the full
translate → evaluate loop against locally served models.

This package is the translation half only; evaluation runs through the standard
`gaia2-runner` path. See [`RUNBOOK.md`](../runner/examples/cluster/RUNBOOK.md) for the end-to-end
cluster runbook.

For the architecture — pipeline stages, the TermTable cross-stage contract, cost
model, design rationale and known caveats — see [`DESIGN.md`](DESIGN.md).

## Install

```bash
uv venv -p 3.12 ~/.venvs/omnigaia-mt
source ~/.venvs/omnigaia-mt/bin/activate
uv pip install -e gaia2-cli/mt
```

## Translate

The pipeline talks to an OpenAI-compatible endpoint. Point it at the same local
vLLM servers the runbook stands up:

```bash
export OMNIGAIA_LLM_BASE_URL=http://localhost:8000/v1
export OMNIGAIA_LLM_API_KEY=EMPTY        # vLLM accepts any non-empty value

# Convenience wrapper (env-overridable): produces a dataset_root on /checkpoint.
TGT_LANG=spa_Latn SUBSET=search LIMIT=10 gaia2-cli/mt/scripts/run_translate.sh
```

Or call the CLI directly:

```bash
python -m omnigaia.cli.translate \
    --output_dir /path/to/shared/omnigaia-mt/spa_Latn/data \
    --dataset_id meta-agents-research-environments/gaia2 \
    --subset all \
    --tgt_lang spa_Latn \
    --translation_model gemma-4-31b \
    --review_model gpt-oss-120b \
    --lid_check
```

Output layout (consumed verbatim as `[target].dataset_root`):

```
<output_dir>/
├── search/        scenario_0000.json, scenario_0001.json, ...
├── execution/
├── ambiguity/
└── adaptability/
```

### LLM endpoint resolution

`omnigaia/llm/config.py` resolves the base URL in this order:

1. `OMNIGAIA_LLM_BASE_URL` — when set, **all** models route here (vLLM / any
   OpenAI-compatible server). Pair with `OMNIGAIA_LLM_API_KEY`.
2. Otherwise the internal Llama-API routing table by model-name prefix
   (`gpt-` / `claude-` / `gemini-`), with the key from `LLAMA_API_KEY` or
   `~/.llama_api_key`. This path only works inside the Meta network.

## Evaluate

Point the runner at the translated `dataset_root` and select the OmniGAIA judge
prompts via `[judge].prompt_version`:

```bash
gaia2-runner run-config \
    --config gaia2-cli/runner/examples/cluster/omnigaia_mt.toml
```

See [`runner/examples/cluster/omnigaia_mt.toml`](../runner/examples/cluster/omnigaia_mt.toml).

## Data converters

`scripts/data/json_to_parquet.py` and `scripts/data/parquet_to_json.py` convert
between the per-scenario JSON layout and HuggingFace-style parquet shards (one
`data` column), for sharing datasets or re-importing `manifold getr` dumps.

## Layout

```
mt/
├── omnigaia/
│   ├── cli/translate.py        # fire CLI entry point
│   ├── translation/            # translate / review / orchestrate
│   ├── llm/                    # OpenAI-compatible client + endpoint routing
│   ├── data/                   # GAIA2 JSON parsing, models, app-state
│   ├── prompts/                # translation prompt strings
│   ├── lid.py                  # GlotLID language-ID check (informational)
│   ├── checkpoint.py, reporting.py
│   └── tests/
├── scripts/
│   ├── run_translate.sh        # translation wrapper (env-overridable, vLLM)
│   └── data/                   # json ⇄ parquet converters
└── pyproject.toml
```

## Notes

- The OmniGAIA *judge* prompt overrides live in the gaia2-cli core package
  (`gaia2_core/judge/{omnigaia_prompts,prompt_overrides}.py`), not here — the
  judge runs inside the container. This package only produces the translated
  data; the eval and judging are entirely the gaia2-cli runner's job.
- Universe/persona generation is *not* included — it is not needed for
  translate + evaluate.
