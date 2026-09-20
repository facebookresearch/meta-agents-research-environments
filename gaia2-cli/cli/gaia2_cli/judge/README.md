# Semantic judge backends

The judge compares agent events with oracle events using Python checks for
arguments, dependencies and timing, followed by semantic checks for content.
Semantic checks can use a text model through LiteLLM or TypeSafe's Jev. Changing
the semantic backend does not replace the graph-matching algorithm, hard
checkers, placeholder checker, or exact-content shortcut.

## Select Jev

In a `gaia2-runner run-config` TOML file, replace the `[judge]` section:

```toml
[judge]
provider = "typesafe"
model = "jev-latest"
api_key_env = "TYPESAFE_API_KEY"
extra_body = { threshold = 0.5, timeout = 30.0 }
```

Use your existing TypeSafe API key in the host environment. The runner passes
it as the judge credential to the container. No Hugging Face token or separate
text-generation model is needed by this backend. The agent's own provider and
credentials remain separately configured.

A complete example is
[`quickstart_hermes_jev.toml`](../../../runner/examples/quickstart_hermes_jev.toml).
Rebuild the runtime image to include this implementation before running it.

```sh
# From gaia2-cli/, with TYPESAFE_API_KEY and the agent's API key set:
make gaia2-hermes
uv run --project runner --python 3.12 gaia2-runner run-config \
    --config runner/examples/quickstart_hermes_jev.toml
```

The optional `base_url` is a TypeSafe-compatible API base ending in `/v1`;
the adapter appends `/systemone`. The default is `https://api.typesafe.ai/v1`.
`extra_body` accepts only `threshold` and `timeout` for this provider. Other
providers continue forwarding it to LiteLLM unchanged. The TypeSafe adapter
uses Python's standard HTTP library and requires no additional SDK.

## What Jev evaluates

Each existing semantic checker keeps its rubric, examples, and rendered
candidate/oracle comparison. Those messages become TypeSafe's `state` together
with the original task context. A Noul question asks whether the rubric's
passing verdict is warranted. There is no fabricated textual reasoning and
no parsing of generated success/failure tags.

Jev does not generate subtask text. Its original-task context is provided
directly alongside the oracle comparison; it does not claim to reproduce the
legacy ARE text-extraction pipeline. Existing LiteLLM prompt rendering is
unchanged. Prompt-version overrides, including Omnilingual-GAIA2 rubrics, apply
to both backends.

A vote passes when `noul >= threshold`. Noul is the probability of the passing
verdict, **not** a partial-credit task score. The default threshold is 0.5;
0.5 itself therefore votes pass. The existing checker vote aggregation and
scenario pass/fail rules remain unchanged. API timeouts, HTTP errors and
invalid responses raise instead of silently passing, falling back to another
model, or being treated as semantic failures. HTTP 429 uses the existing
`RateLimitError` signal. Requests have a bounded timeout and are not retried
inside this adapter.

Every completed TypeSafe vote is appended to `judge_decisions.jsonl` in the
scenario state directory. The runner copies it to the scenario's output
directory. Records contain the requested and returned model IDs, probability,
threshold, verdict and a SHA-256 hash of the request body. They contain neither
credentials nor fabricated explanations. The existing failed-comparison
`judge_output` also includes the structured vote records.

## Add another backend

`gaia2_core.judge.backends` defines two transport-independent protocols:

- `SoftChecker`: accepts the checker's input fields, returns `bool | None`, and
  exposes `last_response` for diagnostic output. `full_task` is available as
  additional context. Rubric templates decide which fields to render.
- `SoftCheckerFactory`: receives the rubric template, vote count and existing
  success/failure labels, and returns a `SoftChecker`.

Supply a factory directly to `Judge(..., checker_factory=my_factory)`. The
core imports no provider SDKs and has no knowledge of Jev. Construction errors
from an explicit factory propagate; they cannot disable semantic judging.
`engine=` remains supported for existing callers and is mutually exclusive
with `checker_factory=`.

The CLI's `gaia2_cli.judge.backends.create_checker_factory` selects the
transport. Add a provider there to expose another factory through runner
configuration. TypeSafe's implementation lives entirely in `typesafe.py`.
No action matching or scoring changes are required to add a provider.

## Comparability and validation

This is an opt-in alternative judge, not a calibrated replacement for the
reference judge. Before comparing scores, grade the same frozen traces with
both judges and manually review disagreements. Fix the rubric version, model
version and threshold before measuring an agent change. `jev-latest` can move;
use a versioned model ID supported by your TypeSafe account for reproducible
runs, and retain the returned model ID from the audit records.

The tests exercise HTTP request/response handling with local fixtures and
run the real judge against synthetic oracle/agent events. They do not establish
agreement with the reference judge, vendor latency/cost claims, or live Jev
accuracy on GAIA2.

See [TypeSafe Noul](https://docs.typesafe.ai/primitives/noul) and the
[API quickstart](https://docs.typesafe.ai/introduction/quickstart) for the wire
format and probability interpretation.
