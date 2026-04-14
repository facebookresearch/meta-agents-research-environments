# Trace File Format (`trace.jsonl`)

The trace viewer (`trace_viewer.py`) reads `trace.jsonl` from each scenario's
artifact directory. Each line is a JSON object representing one LLM API call.

## File Location

Written inside the container at `$GAIA2_TRACE_FILE` (typically
`/tmp/trace.jsonl`), then extracted by the eval runner to
`{output_dir}/{scenario_id}/trace.jsonl`.

## Supported Formats

The trace viewer auto-detects three formats:

### Format A — Raw Response (recommended for new integrations)

Each entry has a `raw_response` string containing the raw HTTP response body.
The viewer parses it at render time into structured content.

```jsonl
{
  "seq": 1,
  "timestamp": "2025-03-17T12:00:00.000Z",
  "type": "llm_call",
  "url": "https://api.example.com/v1/chat/completions",
  "latency_ms": 1234,
  "http_status": 200,
  "request": {
    "model": "claude-opus-4-6",
    "messages": [...],
    "tools": [...],
    "max_tokens": 25000
  },
  "raw_response": "{\"id\":\"chatcmpl-...\",\"choices\":[{\"message\":{\"content\":\"...\",\"tool_calls\":[...]},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50}}"
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `seq` | int | yes | 1-based sequence number |
| `timestamp` | string | yes | ISO 8601 timestamp |
| `type` | string | yes | Always `"llm_call"` |
| `url` | string | yes | Upstream API URL (used for provider detection) |
| `latency_ms` | int | yes | Request duration in milliseconds |
| `http_status` | int | yes | HTTP response status code |
| `request` | object | yes | Parsed request body (JSON) |
| `raw_response` | string | yes | Raw response body (SSE or JSON) |

**Supported `raw_response` formats:**

The viewer auto-detects these response formats in `raw_response`:

1. **Anthropic SSE** — `data: {"type":"message_start",...}` lines
2. **OpenAI chat-completions SSE** — `data: {"choices":[{"delta":...}]}` lines
3. **OpenAI Responses SSE** — `data: {"type":"response.output_item.added",...}` lines
4. **OpenAI chat-completions JSON** — `{"choices":[{"message":{"content":"...","tool_calls":[...]}}]}`
5. **OpenAI Responses JSON** — `{"output":[{"type":"message",...},{"type":"function_call",...}]}`
6. **Anthropic JSON** — `{"content":[{"type":"text","text":"..."}],"stop_reason":"end_turn"}`
7. **Google JSON / SSE** — `{"candidates":[...]}` or `streamGenerateContent` chunks
8. **Provider-native JSON** — `{"completion_message": ...}` style payloads

### Format B — Pre-parsed Response

Each entry has a `response` dict with structured content (already parsed).
Used when the trace writer has already parsed the LLM response.

```jsonl
{
  "seq": 1,
  "timestamp": "2025-03-17T12:00:00.000Z",
  "type": "llm_call",
  "request": { "model": "...", "messages": [...] },
  "response": {
    "content": "The answer is 42",
    "tool_calls": [],
    "usage": { "prompt_tokens": 100, "completion_tokens": 50 },
    "model": "claude-opus-4-6",
    "finish_reason": "stop"
  }
}
```

When `response` is a dict, the viewer uses it directly (no parsing needed).

### Format C — No Trace

If `trace.jsonl` doesn't exist, the viewer still generates `trace.html`
from `events.jsonl` (tool calls) and `result.json` (judge results),
but skips the LLM call timeline.

## Who Writes the Trace

| Container | Writer | Location |
|-----------|--------|----------|
| OpenClaw (`gaia2-oc`) | `containers/openclaw/launch.mjs` | Format A with raw HTTP bodies |
| Hermes (`gaia2-hermes`) | `containers/hermes/hermes_worker.py` | Format A with raw HTTP bodies |

Both read `GAIA2_TRACE_FILE`. The eval runner sets it to `/tmp/trace.jsonl`
and extracts it after the scenario completes.

## Trace Viewer Features

The viewer renders from the trace:

- **LLM call timeline**: request/response cards with collapsible content
- **Token usage**: input/output/cache tokens per call and cumulative
- **Tool calls**: extracted from response `tool_calls` arrays
- **Reasoning / thinking blocks**: collapsed by default when the provider emits them
- **Model info**: extracted from response `model` field
- **Latency**: per-call and cumulative timing
- **Provider detection**: from `url` field and parsed payload shape
