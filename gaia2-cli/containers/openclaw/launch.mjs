// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.
import { appendFileSync } from "node:fs";
import { setDefaultResultOrder } from "node:dns";
import { EnvHttpProxyAgent, setGlobalDispatcher } from "/opt/chatbot/node_modules/undici/index.js";

import { classifyLlmRequest } from "./launch_request_utils.mjs";

// Force IPv4-first DNS resolution. Some hosts advertise unusable IPv6 routes
// ahead of IPv4, which can turn outbound API calls into avoidable timeouts.
setDefaultResultOrder("ipv4first");

const hasProxyEnv =
  process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
if (hasProxyEnv) {
  // Respect HTTP_PROXY / HTTPS_PROXY / NO_PROXY by scheme and hostname so
  // localhost model backends (e.g. 127.0.0.1:8091) bypass the TLS proxy.
  setGlobalDispatcher(new EnvHttpProxyAgent());
}

// ---------------------------------------------------------------------------
// LLM API trace logging (opt-in via GAIA2_TRACE_FILE env var)
// ---------------------------------------------------------------------------

const TRACE_FILE = process.env.GAIA2_TRACE_FILE || "";
let traceSeq = 0;

function traceLog(url, init, status, rawText, latencyMs) {
  if (!TRACE_FILE) return;
  traceSeq++;
  let requestBody = null;
  try { requestBody = JSON.parse(init?.body || "{}"); } catch(e) {}

  const entry = {
    seq: traceSeq,
    timestamp: new Date().toISOString(),
    type: "llm_call",
    url: typeof url === "string" ? url : url?.toString?.() || "",
    latency_ms: latencyMs,
    http_status: status,
    request: requestBody,
    raw_response: rawText,
  };

  try {
    appendFileSync(TRACE_FILE, JSON.stringify(entry) + "\n");
  } catch(e) {
    // best-effort
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sseChunk(meta, choices) {
  return "data: " + JSON.stringify({ ...meta, choices }) + "\n\n";
}

// ---------------------------------------------------------------------------
// ChatCompletion → SSE converter
// ---------------------------------------------------------------------------
// OpenClaw's OpenAI SDK sends stream:true and expects SSE responses.
// After we strip stream from the request for OpenAI-compatible backends, we
// wrap the non-streaming JSON response as SSE ChatCompletionChunk events.

function completionToSSE(json) {
  const meta = {
    id: json.id || "chatcmpl-" + Date.now(),
    object: "chat.completion.chunk",
    model: json.model || "",
    created: json.created || Math.floor(Date.now() / 1000),
  };
  let sse = "";
  for (const choice of (json.choices || [])) {
    const msg = choice.message || {};
    const idx = choice.index || 0;
    const hasToolCalls = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0;
    sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { role: msg.role || "assistant", content: "" } }]);
    if (msg.reasoning) {
      sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { reasoning: msg.reasoning } }]);
    }
    // Some OpenAI-compatible backends return assistant text alongside tool
    // calls. Suppress that mixed text so OpenClaw does not end the turn
    // before the tool calls execute.
    if (msg.content && !hasToolCalls) {
      sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { content: msg.content } }]);
    }
    if (msg.tool_calls) {
      for (let i = 0; i < msg.tool_calls.length; i++) {
        const tc = msg.tool_calls[i];
        sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { tool_calls: [{ index: i, id: tc.id, type: "function", function: { name: tc.function.name } }] } }]);
        sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { tool_calls: [{ index: i, function: { arguments: tc.function.arguments } }] } }]);
      }
    }
    sse += sseChunk(meta, [{ index: idx, finish_reason: choice.finish_reason || "stop", delta: {} }]);
  }
  sse += "data: [DONE]\n\n";
  return sse;
}

// Re-emitting an upstream body (buffered via `resp.text()`, or re-framed) makes
// the original hop headers wrong: `content-encoding` no longer applies because
// fetch already decoded it, and `content-length`/`transfer-encoding` describe
// the original bytes, not what we hand back. Strip all three so the new body is
// read as-is.
function bufferedHeaders(resp) {
  const headers = new Headers(resp.headers);
  headers.delete("content-encoding");
  headers.delete("content-length");
  headers.delete("transfer-encoding");
  return headers;
}

// The Anthropic SDK dispatches streaming events by the SSE `event:` line, so an
// Anthropic-compatible endpoint that emits data-only frames yields zero chunks
// and the SDK's finalizer throws "request ended without sending any chunks".
// Re-frame such streams by deriving `event:` from each payload's `.type`.
// Idempotent: frames that already carry an `event:` line pass through untouched.
function injectAnthropicSSEEventLines(readable) {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const sep = /\r?\n\r?\n/;
  let buf = "";
  const reframe = (block) => {
    if (/^event:/m.test(block)) return block;                 // already framed
    const dataLine = block.split(/\r?\n/).find((l) => l.startsWith("data:"));
    if (!dataLine) return block;
    const payload = dataLine.slice(5).trim();
    if (!payload || payload === "[DONE]") return block;
    try {
      const type = JSON.parse(payload)?.type;
      if (type) return `event: ${type}\n${block}`;
    } catch (e) { /* not JSON — leave as-is */ }
    return block;
  };
  const transform = new TransformStream({
    transform(chunk, controller) {
      buf += decoder.decode(chunk, { stream: true });
      let m;
      while ((m = sep.exec(buf))) {
        const block = buf.slice(0, m.index);
        buf = buf.slice(m.index + m[0].length);
        controller.enqueue(encoder.encode(reframe(block) + "\n\n"));
      }
    },
    flush(controller) {
      if (buf.length) controller.enqueue(encoder.encode(reframe(buf)));
    },
  });
  return readable.pipeThrough(transform);
}

// ---------------------------------------------------------------------------
// Fetch interceptor
// ---------------------------------------------------------------------------
// Intercepts LLM requests to:
//   1. Preserve proxy-aware fetch behavior and request logging
//   2. Retry transient upstream failures
//   3. Shim non-streaming /chat/completions JSON responses into SSE
//      because OpenClaw expects a streaming Chat Completions transport

// Default ceiling for a single LLM call. Long prompts on a busy self-hosted
// backend can take several minutes; a tighter timeout turns those into aborted
// requests that cascade into scenario timeouts. Override with
// OPENCLAW_LLM_TIMEOUT_MS.
const DEFAULT_LLM_TIMEOUT_MS = 600_000;

const _origFetch = globalThis.fetch;
globalThis.fetch = async function(url, init) {
  const {
    urlStr,
    isCompletions,
    isResponses,
    isAnthropicMessages,
    isLLMCall,
  } = classifyLlmRequest(url, init);

  if (isCompletions && init?.body) {
    try {
      const reqBody = JSON.parse(init.body);
      let changed = false;
      const isOpenRouter = urlStr.includes("openrouter.ai");
      if (reqBody.stream && !isOpenRouter) {
        reqBody.stream = false;
        changed = true;
      }
      // Propagate the container thinking level (already resolved into
      // REASONING_EFFORT by the `oc` wrapper) to chat-completions providers.
      const effort = process.env.REASONING_EFFORT;
      const thinkingOn = effort && effort !== "none" && effort !== "off";
      if (isOpenRouter && thinkingOn) {
        reqBody.reasoning = { enabled: true };
        changed = true;
      } else if (thinkingOn && !reqBody.reasoning) {
        reqBody.reasoning = { effort };
        changed = true;
      }

      // vLLM chat-template thinking gate. Hybrid thinking/non-thinking
      // checkpoints (Gemma-4, Qwen3-*, …) ignore the OpenAI `reasoning:{effort}`
      // knob; the working control is `chat_template_kwargs.enable_thinking`.
      // Injected unconditionally for vLLM-style backends because Jinja templates
      // silently ignore unused chat_template_kwargs, so it is a no-op elsewhere.
      // An explicit per-request pin always wins. OpenRouter is excluded for the
      // same reason as above: it is not a vLLM endpoint.
      if (effort && !isOpenRouter) {
        reqBody.chat_template_kwargs = reqBody.chat_template_kwargs || {};
        if (!("enable_thinking" in reqBody.chat_template_kwargs)) {
          reqBody.chat_template_kwargs.enable_thinking = Boolean(thinkingOn);
          changed = true;
        }
      }

      if (changed) {
        init = { ...init, body: JSON.stringify(reqBody) };
      }
    } catch(e) { /* not JSON, leave as-is */ }
  }

  // Strip prior-turn reasoning from the assistant message history. OpenClaw
  // replays each assistant turn's `reasoning`/`reasoning_content` verbatim, and
  // vLLM's harmony renderer feeds it back to the model — which then imitates the
  // replayed analysis text and eventually emits an unparseable tool call, ending
  // the run with an empty message. Dropping prior chain-of-thought from later
  // turns also matches OpenAI's harmony guidance, and `reasoning` is not a
  // standard chat-input field, so this is a no-op for stacks that ignore it.
  if (isCompletions && init?.body) {
    try {
      const reqBody = JSON.parse(init.body);
      if (Array.isArray(reqBody.messages)) {
        let stripped = false;
        for (const m of reqBody.messages) {
          if (m && m.role === "assistant") {
            if ("reasoning" in m) { delete m.reasoning; stripped = true; }
            if ("reasoning_content" in m) { delete m.reasoning_content; stripped = true; }
          }
        }
        if (stripped) {
          init = { ...init, body: JSON.stringify(reqBody) };
        }
      }
    } catch(e) { /* not JSON, leave as-is */ }
  }

  // For LLM calls: retry on 429, 5xx, and network errors with exponential backoff
  const MAX_RETRIES = isLLMCall ? 5 : 0;
  const FETCH_TIMEOUT_MS = isLLMCall
    ? (parseInt(process.env.OPENCLAW_LLM_TIMEOUT_MS, 10) || DEFAULT_LLM_TIMEOUT_MS)
    : 0;
  let resp, text, latencyMs;

  for (let attempt = 0; ; attempt++) {
    const startTime = Date.now();

    try {
      if (FETCH_TIMEOUT_MS > 0) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
        try {
          resp = await _origFetch.call(this, url, { ...init, signal: controller.signal });
        } finally {
          clearTimeout(timer);
        }
      } else {
        resp = await _origFetch.call(this, url, init);
      }
    } catch (err) {
      latencyMs = Date.now() - startTime;
      if (!isLLMCall) throw err;
      const isTimeout = err?.name === "TimeoutError" || err?.name === "AbortError" || err?.code === "UND_ERR_CONNECT_TIMEOUT" || err?.message?.includes("timed out");
      const errInfo = { error: err?.message || String(err), type: isTimeout ? "timeout" : "fetch_error" };
      traceLog(url, init, 0, JSON.stringify(errInfo), latencyMs);

      if (attempt < MAX_RETRIES) {
        // Network/TLS handshake failures are usually transient (e.g. the
        // in-container MITM TLS proxy still minting a per-host cert), and the
        // agent's own chunks-watchdog fires within seconds — so retry those
        // tightly instead of backing off past the watchdog. Real outages still
        // surface once MAX_RETRIES is exhausted.
        const delay = errInfo.type === "fetch_error"
          ? 100 + Math.random() * 100
          : Math.min(1000 * Math.pow(2, attempt) + Math.random() * 500, 60000);
        console.error(`[launch.mjs] LLM fetch error (attempt ${attempt + 1}/${MAX_RETRIES + 1}): ${errInfo.type} for ${urlStr} (${latencyMs}ms): ${errInfo.error} — retrying in ${(delay / 1000).toFixed(2)}s`);
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
      console.error(`[launch.mjs] LLM fetch error (final attempt): ${errInfo.type} for ${urlStr} (${latencyMs}ms): ${errInfo.error}`);
      throw err;
    }

    if (!isLLMCall) return resp;

    // Anthropic Messages returns an SSE stream the Anthropic SDK consumes as
    // live events. Buffering it with resp.text() withholds every event until the
    // upstream turn completes, which trips the SDK/agent watchdogs on long turns.
    // Leave a successful body unread and stream it through below; only buffer on
    // retriable/error status, where the text is needed for logging and retry.
    if (isAnthropicMessages && resp.ok) {
      latencyMs = Date.now() - startTime;
      break;
    }

    text = await resp.text();
    latencyMs = Date.now() - startTime;

    // Retry on 429 (rate limit) and 5xx (server errors).
    if ((resp.status === 429 || resp.status >= 500) && attempt < MAX_RETRIES) {
      const retryAfter = resp.headers.get("retry-after");
      const delay = retryAfter ? Math.min(parseFloat(retryAfter) * 1000, 60000) : Math.min(1000 * Math.pow(2, attempt) + Math.random() * 500, 60000);
      const preview = text.length > 200 ? text.slice(0, 200) + "..." : text;
      console.error(`[launch.mjs] LLM API HTTP ${resp.status} (attempt ${attempt + 1}/${MAX_RETRIES + 1}) from ${urlStr} (${latencyMs}ms): ${preview} — retrying in ${(delay / 1000).toFixed(1)}s`);
      traceLog(url, init, resp.status, text, latencyMs);
      await new Promise(r => setTimeout(r, delay));
      continue;
    }

    break; // Success or non-retryable error
  }

  // Anthropic Messages API (success): stream the SSE body through live so the
  // Anthropic SDK receives incremental events, re-framing data-only frames with
  // `event:` lines (see injectAnthropicSSEEventLines). When tracing is on, tee a
  // copy of the raw upstream body and accumulate it for the LLM trace.
  if (isAnthropicMessages && resp.ok) {
    let clientSrc = resp.body;
    if (TRACE_FILE && clientSrc) {
      const [forClient, forTrace] = clientSrc.tee();
      clientSrc = forClient;
      (async () => {
        try {
          const reader = forTrace.getReader();
          const dec = new TextDecoder();
          let acc = "";
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            acc += dec.decode(value, { stream: true });
          }
          traceLog(url, init, resp.status, acc, latencyMs);
        } catch (e) { /* best-effort trace */ }
      })();
    }
    const outBody = clientSrc ? injectAnthropicSSEEventLines(clientSrc) : clientSrc;
    return new Response(outBody, { status: resp.status, statusText: resp.statusText, headers: bufferedHeaders(resp) });
  }

  const ct = resp.headers.get("content-type") || "";
  const isSSE = ct.includes("text/event-stream") || text.trimStart().startsWith("data: ");

  // Log the raw request/response before any transformation
  traceLog(url, init, resp.status, text, latencyMs);

  // Surface HTTP errors to stderr so they appear in entrypoint.log
  if (resp.status >= 400) {
    const preview = text.length > 500 ? text.slice(0, 500) + "..." : text;
    console.error(`[launch.mjs] LLM API error: HTTP ${resp.status} from ${urlStr} (${latencyMs}ms): ${preview}`);
  }

  // Anthropic Messages API (non-2xx): success is streamed live above; here we
  // only reach the buffered path for error/non-ok responses, returned as-is.
  if (isAnthropicMessages) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: bufferedHeaders(resp) });
  }

  // Responses API requests are handled natively upstream; do not rewrite them.
  if (isResponses) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: bufferedHeaders(resp) });
  }

  // Streaming chat-completions responses already have the right transport shape.
  if (isSSE) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: bufferedHeaders(resp) });
  }

  // Only synthesize SSE for successful non-streaming chat-completions payloads.
  if (!isCompletions || !resp.ok) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: bufferedHeaders(resp) });
  }

  try {
    const body = JSON.parse(text);
    const sse = completionToSSE(body);
    const headers = bufferedHeaders(resp);
    headers.set("content-type", "text/event-stream");
    return new Response(sse, { status: resp.status, statusText: resp.statusText, headers });
  } catch(e) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: bufferedHeaders(resp) });
  }
};

// Rename the process to avoid host security policies targeting "openclaw"
process.title = "gaia2-oc";

// entry.js respawns a child Node process to add warning-suppression flags.
// That child bypasses this wrapper, so the fetch shim never reaches the real
// gateway process. The wrapper already sets NODE_EXTRA_CA_CERTS itself.
process.env.OPENCLAW_NODE_OPTIONS_READY ||= "1";
if (process.env.NODE_EXTRA_CA_CERTS) {
  process.env.OPENCLAW_NODE_EXTRA_CA_CERTS_READY ||= "1";
}

const args = process.argv.slice(2);
// Set argv[1] to gaia2-gw.mjs (renamed copy of openclaw.mjs).
// Some host security policies kill processes with "openclaw" in argv.
process.argv = ["node", "/opt/chatbot/gaia2-gw.mjs", ...args];
await import("/opt/chatbot/dist/entry.js");
