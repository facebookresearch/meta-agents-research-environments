# gaia2-hermes

`gaia2-hermes` packages the Gaia2 environment around a Hermes worker. It keeps
the same adapter contract as the other runtime images, but the model loop lives
in a dedicated Hermes worker process that talks to the adapter over a Unix
socket.

Most benchmark users should launch it through `gaia2-runner`, not by hand.

## Architecture

```text
host runner
    |
    v
gaia2_adapter.py (:8090, gaia2 user)
    |
    +--> /notify, /messages, /events, /health, /status
    |
    v
Unix socket bridge
    |
    v
hermes_worker.py (agent user)
    |
    v
upstream model API

gaia2-eventd (gaia2 user)
    |
    +--> watches events.jsonl
    +--> triggers ENV reactions
    +--> runs in-container judging
```

Ownership split:

- `root`: initializes state and starts the Gaia2 infrastructure
- `gaia2`: owns `/var/gaia2/state`, the adapter, and the daemon
- `agent`: owns the Hermes worker and the restricted tool shell

The worker never reads raw scenario state directly. It only interacts with the
environment through Gaia2 CLI tools, which go through `gaia2-exec`.

## Build

```bash
cd /path/to/gaia2-cli
make gaia2-hermes
```

## Direct Container Run

This is mainly for debugging the runtime itself. For full evals, use the
runner.

```bash
export SCENARIO=/path/to/scenario.json

podman run --rm --network=host \
  -v "$SCENARIO:/var/gaia2/custom_scenario.json:ro,z" \
  -e PROVIDER=anthropic \
  -e MODEL=claude-opus-4-6 \
  -e API_KEY="$ANTHROPIC_API_KEY" \
  -e THINKING=high \
  localhost/gaia2-hermes:latest
```

As with the other runtimes, starting the container only brings up the adapter
and worker. The task itself still arrives later through `POST /notify`.

## Runtime Configuration

| Env var | Description |
|---------|-------------|
| `PROVIDER` | Provider name such as `anthropic`, `openai`, `openai-compat`, `gemini`, or `openrouter` |
| `MODEL` | Raw model identifier passed into Hermes |
| `API_KEY` | Optional explicit API key |
| `BASE_URL` | Optional custom base URL |
| `THINKING` | Reasoning effort forwarded to Hermes |
| `MAX_TOKENS` | Optional runtime token cap |
| `GAIA2_TRACE_FILE` | Optional raw request/response trace JSONL path |

Provider-specific key env vars also work when exported before container start.
For direct Google AI Studio runs, use `PROVIDER=gemini` and set
`BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`.

## Startup Sequence

At runtime the container does this:

1. `gaia2-init-entrypoint.sh` initializes state and renders the prompt
2. `gaia2-eventd` starts as `gaia2`
3. `gaia2_adapter.py` starts as `gaia2`
4. `entrypoint.sh` starts `hermes_worker.py` as `agent`
5. the adapter and worker connect over a Unix socket

This split is different from OpenClaw: Hermes does not need a TLS MITM proxy.

## Tracing and Faketime

Hermes writes raw traces from the worker side:

- `hermes_worker.py` writes `trace.jsonl` when `GAIA2_TRACE_FILE` is set
- the worker uses a non-streaming path so raw traces stay simple across generic
  providers
- the host trace viewer normalizes those raw payloads at render time

See [runner/TRACE_FORMAT.md](../../runner/TRACE_FORMAT.md) for the trace
contract.

Faketime behavior:

- CLI-facing behavior and simulated time reads follow the scenario clock
- outbound HTTPS calls stay on real time
- unlike OpenClaw, Hermes does not need a local TLS proxy for that split

## HTTP Contract

The adapter exposes the same surface as the other runtimes:

- `POST /notify`
- `GET /messages?after=<seq>`
- `GET /events`
- `GET /health`
- `GET /status`

Manual probe:

```bash
curl -sf http://127.0.0.1:8090/health | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8090/notify \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the date today?"}'
curl -s 'http://127.0.0.1:8090/messages?after=0' | python3 -m json.tool
```

## Debugging

Useful logs:

- `/tmp/entrypoint.log`
- `/tmp/gaia2-adapter.log`
- `/tmp/gaia2-eventd.log`

Useful commands:

```bash
podman exec <container> tail -n 80 /tmp/entrypoint.log
podman exec <container> tail -n 80 /tmp/gaia2-adapter.log
podman exec <container> tail -n 80 /tmp/gaia2-eventd.log
podman exec <container> ls -l /tmp/trace.jsonl
```

If a run finishes but the benchmark fails, inspect `result.json` and
`daemon_status.json` in the extracted artifact directory first. Hermes often
fails on agent behavior, not infrastructure.
