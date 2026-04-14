# gaia2-oracle

`gaia2-oracle` is the no-agent replay image. It does not call a model. Instead
it replays the scenario's ground-truth tool actions through the same Gaia2 CLI
surface and lets the daemon run the same turn logic and judging pipeline used
for agent runs.

Use it when you want:

- a fast end-to-end smoke test of the runner and image build
- a correctness baseline for a scenario
- a way to separate infrastructure problems from model behavior

## Architecture

```text
host runner
    |
    v
gaia2_oracle_adapter.py (:8090, gaia2 user)
    |
    +--> /notify, /health, /status
    |
    v
oracle replay logic
    |
    v
events.jsonl
    |
    v
gaia2-eventd (gaia2 user)
    |
    +--> triggers ENV reactions
    +--> detects turn boundaries
    +--> runs in-container judging
```

Ownership is simpler than the agent runtimes:

- `root`: initializes state and starts the daemon
- `gaia2`: owns the adapter, replay logic, daemon, and `/var/gaia2/state`

There is no separate `agent` process because there is no live model.

## Build

```bash
cd /path/to/gaia2-cli
make gaia2-oracle
```

## Direct Container Run

```bash
export SCENARIO=/path/to/scenario.json

podman run --rm --network=host \
  -v "$SCENARIO:/var/gaia2/custom_scenario.json:ro,z" \
  localhost/gaia2-oracle:latest
```

As with the other images, the container just starts the adapter. The replay
begins when the runner or a manual client sends the initial task through
`POST /notify`.

## Judge Configuration

Oracle does not need agent-side `PROVIDER`, `MODEL`, or `API_KEY`, but the
benchmark still needs judge settings if you want an in-container verdict:

| Env var | Description |
|---------|-------------|
| `GAIA2_JUDGE_PROVIDER` | Judge provider name |
| `GAIA2_JUDGE_MODEL` | Judge model name |
| `GAIA2_JUDGE_BASE_URL` | Optional judge endpoint override |
| `GAIA2_JUDGE_API_KEY` | Optional explicit judge API key |
| `GAIA2_TIME_SPEED` | Optional time acceleration for time scenarios |

In normal benchmark use, the host runner sets these env vars for you.

## Why Oracle Is Useful

Oracle is the best first check after changing the runner, build chain, or
trace viewer:

- if Oracle passes, the core container and judge plumbing are probably fine
- if a live agent fails while Oracle passes on the same scenario, the issue is
  usually model behavior rather than infrastructure

There is no model-side `trace.jsonl` because no LLM requests are made.

## HTTP Contract

The Oracle adapter intentionally looks like the agent adapters from the
runner's point of view:

- `POST /notify`
- `GET /health`
- `GET /status`

Manual probe:

```bash
curl -sf http://127.0.0.1:8090/health | python3 -m json.tool
curl -sf http://127.0.0.1:8090/status | python3 -m json.tool
```

## Notes on Time Scenarios

Time scenarios still rely on faketime and daemon polling, even in Oracle mode.
That means wall-clock runtime can still be noticeable for the `time` split,
especially when you keep `GAIA2_TIME_SPEED` at its default.

## Debugging

Useful logs:

- `/tmp/entrypoint.log`
- `/tmp/gaia2-eventd.log`

Useful commands:

```bash
podman exec <container> tail -n 80 /tmp/entrypoint.log
podman exec <container> tail -n 80 /tmp/gaia2-eventd.log
curl -sf http://127.0.0.1:8090/health | python3 -m json.tool
curl -sf http://127.0.0.1:8090/status | python3 -m json.tool
```
