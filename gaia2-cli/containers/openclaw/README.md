# gaia2-openclaw

`gaia2-openclaw` packages the Gaia2 environment behind the OpenClaw runtime.
Use it when you want the most configurable container for Anthropic, OpenAI,
Google, OpenRouter, or other compatibility endpoints.

Most benchmark users should launch it through `gaia2-runner`, not with
`podman run` directly.

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
OpenClaw gateway (:18789, agent user)
    |
    v
upstream model API

gaia2-eventd (gaia2 user)
    |
    +--> watches events.jsonl
    +--> triggers ENV reactions
    +--> runs in-container judging

CLI tools
    |
    v
gaia2-exec -> /var/gaia2/state
```

Runtime ownership is split intentionally:

- `root`: initializes scenario state, renders the agent prompt, and starts the
  optional TLS proxy when faketime is active
- `gaia2`: owns `/var/gaia2/state`, the adapter, and the daemon
- `agent`: owns the OpenClaw gateway process and agent-facing shell

## What Is OpenClaw-Specific Here?

OpenClaw is the only runtime in this repo that needs extra transport handling
when simulated time is active:

- the gateway itself runs under libfaketime
- outbound HTTPS is routed through a local MITM TLS proxy so Node can keep
  talking to providers safely
- `launch.mjs` normalizes provider responses and logs raw model traffic

That is why OpenClaw is the richest runtime for compatibility endpoints, but
also the one with the most moving parts.

## Build

```bash
cd /path/to/gaia2-cli
make gaia2-oc
```

That target builds the `gaia2-cli` base first and then
`localhost/gaia2-oc:latest`.

Prefer `localhost/gaia2-oc:latest` for image tags. Some host security tooling
keys off `openclaw` in process or image names and may kill the process tree.

## Direct Container Run

Direct `podman run` is useful for low-level debugging. For normal evals, use
the host-side runner instead.

```bash
export SCENARIO=/path/to/scenario.json

podman run --rm --network=host \
  -v "$SCENARIO:/var/gaia2/custom_scenario.json:ro,z" \
  -e PROVIDER=anthropic \
  -e MODEL=claude-opus-4-6 \
  -e API_KEY="$ANTHROPIC_API_KEY" \
  -e OPENCLAW_FORCE_RECONFIG=1 \
  localhost/gaia2-oc:latest
```

Important:

- `OPENCLAW_FORCE_RECONFIG=1` is required for direct runs because the runtime
  config is regenerated from env vars at startup
- starting the container only brings up the adapter and runtime; the actual
  task still arrives later over `POST /notify`

## HTTP Contract

These are the endpoints the runner talks to:

- `POST /notify` sends the initial user task or a daemon follow-up
- `GET /messages?after=<seq>` polls buffered assistant responses
- `GET /events` streams responses over SSE
- `GET /health` reports connection status to the gateway
- `GET /status` reports daemon progress

Example manual probe:

```bash
curl -sf http://127.0.0.1:8090/health | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8090/notify \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the date today?"}'
curl -s 'http://127.0.0.1:8090/messages?after=0' | python3 -m json.tool
```

## Provider Configuration

Common runtime env vars:

| Env var | Description |
|---------|-------------|
| `PROVIDER` | `anthropic`, `openai`, `openai-compat`, `openai-completions`, `google`, or `openrouter` |
| `MODEL` | Model identifier passed through to the backend |
| `API_KEY` | Explicit API key override |
| `BASE_URL` | Optional custom endpoint override |
| `THINKING` | Reasoning/thinking level |
| `OPENCLAW_FORCE_RECONFIG` | Required for direct runs |
| `GAIA2_TRACE_FILE` | Optional raw trace JSONL output |

Provider-specific behavior:

- `anthropic`: direct Anthropic Messages, or `anthropic-messages` when
  `BASE_URL` or `ANTHROPIC_BASE_URL` is set
- `openai`: direct OpenAI Responses, or `openai-responses` when
  `BASE_URL` or `OPENAI_BASE_URL` is set
- `openai-compat` and `openai-completions`: arbitrary OpenAI
  chat-completions-compatible endpoints using the raw model name unchanged
- `google`: direct Google support, or `google-generative-ai` with a custom base
  URL
- `openrouter`: routed through an OpenAI-completions-style provider config

If you are onboarding a new external endpoint:

- use `PROVIDER=openai` for Responses-style APIs
- use `PROVIDER=openai-compat` for generic chat-completions APIs

## Startup Sequence

At runtime the container does this:

1. `gaia2-init-entrypoint.sh` initializes scenario state and renders the prompt
2. `gaia2-eventd` starts as `gaia2`
3. `gaia2_adapter.py` starts as `gaia2`
4. optional `tls_proxy.py` starts as `root` when faketime is active
5. `entrypoint.sh` starts the OpenClaw gateway as `agent`
6. the gateway is health-checked before the container is considered ready

## Tracing and Faketime

OpenClaw emits the richest raw model traces in the repo:

- `launch.mjs` writes `trace.jsonl` when `GAIA2_TRACE_FILE` is set
- the trace keeps raw provider responses intact
- the host trace viewer normalizes those raw payloads at render time

See [runner/TRACE_FORMAT.md](../../runner/TRACE_FORMAT.md) for the exact trace
contract.

Faketime model:

- the gateway sees simulated wall-clock time
- CLI tools also see simulated time through `gaia2-exec` and shell wrappers
- outbound HTTPS is kept on real time through the TLS proxy path

## Debugging

Useful logs inside the container:

- `/tmp/entrypoint.log`
- `/tmp/gaia2-adapter.log`
- `/tmp/gaia2-eventd.log`
- `/tmp/openclaw/*.log`

Useful commands:

```bash
podman exec <container> tail -n 80 /tmp/entrypoint.log
podman exec <container> tail -n 80 /tmp/gaia2-adapter.log
podman exec <container> tail -n 80 /tmp/gaia2-eventd.log
podman exec <container> ls -l /tmp/trace.jsonl
```

Common pitfalls:

- if the gateway never comes up, inspect `/tmp/entrypoint.log`
- if the agent is healthy but the run never finishes, inspect
  `/tmp/gaia2-eventd.log` for missing turn boundaries
- if outbound HTTPS fails only under faketime, check the TLS proxy path and the
  host-side proxy/CA wiring from the runner
