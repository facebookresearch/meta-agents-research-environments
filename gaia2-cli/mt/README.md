# Omnilingual-GAIA2: multilingual translation pipeline

Omnilingual-GAIA2 translates the Gaia2 benchmark into a target language, producing a
`dataset_root` directory of translated scenario JSON files that the
[gaia2-cli runner](../runner/README.md) evaluates directly. Paired with the
`omnilingual-gaia2` judge prompts (ported into `gaia2_core.judge`), it runs the full
translate → evaluate loop against locally served models.

This package is the translation half only; evaluation runs through the standard
`gaia2-runner` path — see [Evaluate](#evaluate) below.

For the architecture — pipeline stages, the TermTable cross-stage contract, cost
model, design rationale and known caveats — see [`DESIGN.md`](DESIGN.md).

## Install

```bash
uv venv -p 3.12 ~/.venvs/omnilingual-gaia2
source ~/.venvs/omnilingual-gaia2/bin/activate
uv pip install -e gaia2-cli/mt
```

## Translate

The pipeline talks to an OpenAI-compatible endpoint. Point it at the same local
vLLM servers the runbook stands up:

```bash
export GAIA2_MT_LLM_BASE_URL=http://localhost:8000/v1
export GAIA2_MT_LLM_API_KEY=EMPTY        # vLLM accepts any non-empty value

# Convenience wrapper (env-overridable); set OUTPUT_BASE to your shared scratch.
TGT_LANG=spa_Latn SUBSET=search LIMIT=10 gaia2-cli/mt/scripts/run_translate.sh
```

Or call the CLI directly:

```bash
python -m gaia2_mt.cli.translate \
    --output_dir /path/to/shared/omnilingual-gaia2/spa_Latn/data \
    --dataset_id meta-agents-research-environments/gaia2 \
    --subset all \
    --tgt_lang spa_Latn \
    --translation_model google/gemma-4-31B-it
```

The defaults reproduce the released dataset: one translator-only pass with the
term-table contract enforced. Add `--review` for the optional second
review + post-edit pass, and `--results_csv PATH` for the per-scenario audit
trail.

Output layout (consumed verbatim as `[target].dataset_root`):

```
<output_dir>/
├── search/        scenario_0000.json, scenario_0001.json, ...
├── execution/
├── ambiguity/
└── adaptability/
```

### LLM endpoint resolution

`gaia2_mt/llm/config.py` resolves the base URL in this order:

1. `GAIA2_MT_PER_MODEL_ENDPOINTS` — JSON `{model_name: base_url}`, for serving
   the translator and reviewer on separate ports.
2. `GAIA2_MT_LLM_BASE_URL` — one endpoint for every model.
3. Otherwise `http://localhost:8000/v1`.

Pair either with `GAIA2_MT_LLM_API_KEY`; vLLM accepts any non-empty value. Model
names must match the server's `--served-model-name`.

## Evaluate

Either way, select the Omnilingual-GAIA2 judge prompts via
`[judge].prompt_version = "omnilingual-gaia2"`, then:

```bash
gaia2-runner run-config \
    --config gaia2-cli/runner/examples/openclaw_qwen_omnilingual_gaia2_pass3.toml
```

For the **published** dataset there is nothing to download by hand — set
`[target].language` and the runner fetches and caches the per-language HuggingFace
config itself:

```toml
[target]
dataset = "facebook/omnilingual-gaia2"
language = "spa_Latn"
```

For a corpus **you** just translated with the pipeline above, point at its output
directory instead:

```toml
[target]
dataset_root = "/path/to/omnilingual-gaia2/spa_Latn/data"
splits = ["execution", "search", "ambiguity", "adaptability"]
```

See [`runner/examples/openclaw_qwen_omnilingual_gaia2_pass3.toml`](../runner/examples/openclaw_qwen_omnilingual_gaia2_pass3.toml).

## Data converters

`scripts/data/json_to_parquet.py` and `scripts/data/parquet_to_json.py` convert
between the per-scenario JSON layout and HuggingFace-style parquet shards (one
`data` column), for sharing datasets or re-importing `manifold getr` dumps.

## Layout

```
mt/
├── gaia2_mt/
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

- The Omnilingual-GAIA2 *judge* prompt overrides live in the gaia2-cli core package
  (`gaia2_core/judge/{omnilingual_gaia2_prompts,prompt_overrides}.py`), not here — the
  judge runs inside the container. This package only produces the translated
  data; the eval and judging are entirely the gaia2-cli runner's job.
- Universe/persona generation is *not* included — it is not needed for
  translate + evaluate.
