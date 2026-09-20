# TypeSafe Jev judge

Set the existing judge provider to `typesafe` and export `TYPESAFE_API_KEY`:

```toml
[judge]
provider = "typesafe"
model = "jev-latest"
api_key_env = "TYPESAFE_API_KEY"
extra_body = { threshold = 0.5, timeout = 30.0 }
```

`base_url` optionally overrides `https://api.typesafe.ai/v1`. Use a supported
versioned model ID for reproducible comparisons. The quickstart example is
[`quickstart_hermes_jev.toml`](../../../runner/examples/quickstart_hermes_jev.toml).

The adapter sends the existing rendered rubric, examples and candidate to
`POST /systemone` as a typed Noul question. It converts the returned probability
into the rubric's existing success/failure tag using `probability >= threshold`.
GAIA2's current checkers still perform voting and record the response. The
response also includes the requested/returned model, probability, threshold and
input hash. These tags are adapter output, not model-generated explanations.
HTTP failures and malformed responses raise errors rather than becoming votes.

No core judge, prompt, subtask, hard-check, ordering or timing logic is changed.
Jev receives the same rendered inputs as the existing engine. The adapter only
supports binary checker prompts with the standard GAIA2 verdict tags; arbitrary
text generation, custom tag formats and subtask extraction are unsupported.
Legacy ARE integrations must retain a text engine for subtask extraction.

Other Python integrations can import `TypeSafeEngine` from
`gaia2_cli.judge.typesafe`. It implements the existing callable engine interface,
`engine(messages, **kwargs) -> (text, metadata)`, and accepts an optional
`audit_path` for JSONL records. It does not require LiteLLM or a new SDK.

Jev is an alternative judge. Its scores are not calibrated against the reference
Llama judge. Report the provider, returned model and threshold with results and
compare saved traces before treating scores as interchangeable. Automated tests
use mocked API responses and local HTTP; no live accuracy claim is made.
