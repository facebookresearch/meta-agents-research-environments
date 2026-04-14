#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Hermes agent worker — runs as the sandboxed agent user.

Connects to the GAIA2 adapter (gaia2 user) via a Unix socket and processes
conversation requests using the Hermes AIAgent.

Protocol (JSON lines over Unix socket):
    Adapter → Worker:
        {"type": "message", "text": "...", "run_id": "..."}
        {"type": "interrupt", "text": "..."}
    Worker → Adapter:
        {"type": "ready"}
        {"type": "response", "run_id": "...", "state": "final"|"error", "message": "..."}
"""

import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any

WORKER_SOCK = os.environ.get("HERMES_WORKER_SOCK", "/tmp/hermes-worker.sock")
TRACE_FILE = os.environ.get("GAIA2_TRACE_FILE", "")
CONNECT_TIMEOUT_SECONDS = float(os.environ.get("HERMES_WORKER_CONNECT_TIMEOUT", "60"))
CONNECT_RETRY_INTERVAL_SECONDS = float(
    os.environ.get("HERMES_WORKER_CONNECT_RETRY_INTERVAL", "0.2")
)
# Rendered at container startup by gaia2-init-entrypoint.sh from AGENTS_TEMPLATE.md
AGENTS_MD = os.path.expanduser("~/AGENTS.md")
_trace_lock = threading.Lock()
_trace_depth = threading.local()
_trace_seq = 0


# ═══════════════════════════════════════════════════════════════════════
#  AIAgent
# ═══════════════════════════════════════════════════════════════════════


def _patch_hermes_time():
    """Override hermes_time.now() to return simulated time from faketime.rc."""
    from datetime import datetime, timezone

    import hermes_time

    _original_now = hermes_time.now

    def _fake_now():
        try:
            with open("/tmp/faketime.rc") as f:
                ts = f.read().strip()
            if ts:
                return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
        except (OSError, ValueError):
            pass
        return _original_now()

    hermes_time.now = _fake_now


def _strip_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_tool_call_object(
    tool_call: Any, normalized_id: str | None = None
) -> None:
    """Normalize provider-returned tool call IDs in place when possible.

    Some OpenAI-compatible providers pad tool call IDs with whitespace. Hermes
    strips that whitespace when storing assistant messages, but later reuses the
    raw provider object when appending role="tool" results. That mismatch causes
    the next-turn sanitizer to discard the real tool outputs as "orphaned".
    """

    call_id = _strip_optional_string(normalized_id)
    if call_id is None:
        call_id = _strip_optional_string(getattr(tool_call, "id", None))
    if call_id is not None:
        try:
            setattr(tool_call, "id", call_id)
        except Exception:
            pass
        try:
            if hasattr(tool_call, "call_id") or getattr(tool_call, "call_id", None):
                setattr(tool_call, "call_id", call_id)
        except Exception:
            pass

    response_item_id = _strip_optional_string(
        getattr(tool_call, "response_item_id", None)
    )
    if response_item_id is not None:
        try:
            setattr(tool_call, "response_item_id", response_item_id)
        except Exception:
            pass


def _normalize_stored_tool_call_ids(messages: list[dict[str, Any]]) -> None:
    """Normalize tool-call identifiers in persisted chat history."""

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                for key in ("id", "call_id", "response_item_id"):
                    value = _strip_optional_string(tool_call.get(key))
                    if value is not None:
                        tool_call[key] = value

        if msg.get("role") == "tool":
            tool_call_id = _strip_optional_string(msg.get("tool_call_id"))
            if tool_call_id is not None:
                msg["tool_call_id"] = tool_call_id


def _patch_hermes_tool_call_ids(agent_cls: type) -> None:
    """Patch Hermes to keep tool-call IDs stable across provider quirks."""

    if getattr(agent_cls, "_gaia2_tool_call_id_patch_installed", False):
        return

    original_build_assistant_message = agent_cls._build_assistant_message

    @wraps(original_build_assistant_message)
    def _wrapped_build_assistant_message(self, assistant_message, finish_reason):
        message = original_build_assistant_message(
            self, assistant_message, finish_reason
        )
        normalized_calls = (
            message.get("tool_calls") if isinstance(message, dict) else None
        )

        for idx, tool_call in enumerate(
            getattr(assistant_message, "tool_calls", None) or []
        ):
            normalized_id = None
            if isinstance(normalized_calls, list) and idx < len(normalized_calls):
                normalized_call = normalized_calls[idx]
                if isinstance(normalized_call, dict):
                    normalized_id = normalized_call.get("id")
            _normalize_tool_call_object(tool_call, normalized_id=normalized_id)

        return message

    original_sanitize_api_messages = agent_cls._sanitize_api_messages

    @staticmethod
    @wraps(original_sanitize_api_messages)
    def _wrapped_sanitize_api_messages(messages):
        _normalize_stored_tool_call_ids(messages)
        return original_sanitize_api_messages(messages)

    agent_cls._build_assistant_message = _wrapped_build_assistant_message
    agent_cls._sanitize_api_messages = _wrapped_sanitize_api_messages
    agent_cls._gaia2_tool_call_id_patch_installed = True


def _disable_hermes_streaming_for_gaia2(agent: Any) -> None:
    """Use Hermes' non-streaming path inside Gaia2 containers.

    The Gaia2 adapter only consumes final responses, not incremental tokens.
    For chat-completions-compatible passthrough endpoints, Hermes' streaming
    path adds another transport layer that can fail even when the exact same
    request succeeds non-streaming.
    """

    if getattr(agent, "_gaia2_streaming_disabled", False):
        return

    original_streaming_call = agent._interruptible_streaming_api_call

    @wraps(original_streaming_call)
    def _non_streaming_call(api_kwargs, *args, **kwargs):
        return agent._interruptible_api_call(api_kwargs)

    agent._interruptible_streaming_api_call = _non_streaming_call
    agent._gaia2_streaming_disabled = True


def _normalize_chat_reasoning_request(
    api_kwargs: Any,
    *,
    api_mode: str,
    thinking: str,
) -> Any:
    """Translate Gaia2 thinking into OpenAI chat-completions reasoning control.

    Hermes already exposes a generic ``reasoning_config`` knob, but some
    OpenAI-compatible chat-completions endpoints expect a top-level
    ``reasoning_effort`` field instead. Gaia2 keeps ``thinking`` as the
    user-facing abstraction and normalizes only the wire request here.
    """

    if not isinstance(api_kwargs, dict):
        return api_kwargs
    if api_mode in {"anthropic_messages", "codex_responses"}:
        return api_kwargs
    if "messages" not in api_kwargs:
        return api_kwargs

    normalized = dict(api_kwargs)
    effort = _strip_optional_string(normalized.get("reasoning_effort"))
    changed = False

    extra_body = normalized.get("extra_body")
    if isinstance(extra_body, dict) and "reasoning" in extra_body:
        extra_body_copy = dict(extra_body)
        reasoning = extra_body_copy.pop("reasoning", None)
        if effort is None and isinstance(reasoning, dict):
            effort = _strip_optional_string(reasoning.get("effort"))
        if extra_body_copy:
            normalized["extra_body"] = extra_body_copy
        else:
            normalized.pop("extra_body", None)
        changed = True

    if effort is None:
        effort = _strip_optional_string(thinking)

    if effort is not None and normalized.get("reasoning_effort") != effort:
        normalized["reasoning_effort"] = effort
        changed = True

    return normalized if changed else api_kwargs


def _patch_chat_reasoning_effort(agent: Any, thinking: str) -> None:
    """Inject reasoning_effort for OpenAI-style chat completions requests."""

    normalized_thinking = _strip_optional_string(thinking)
    if normalized_thinking is None:
        return
    if getattr(agent, "_gaia2_reasoning_effort_patch_installed", False):
        return

    original_api_call = agent._interruptible_api_call

    @wraps(original_api_call)
    def _wrapped_api_call(api_kwargs, *args, **kwargs):
        normalized_api_kwargs = _normalize_chat_reasoning_request(
            api_kwargs,
            api_mode=getattr(agent, "api_mode", ""),
            thinking=normalized_thinking,
        )
        return original_api_call(normalized_api_kwargs, *args, **kwargs)

    agent._interruptible_api_call = _wrapped_api_call
    agent._gaia2_reasoning_effort_patch_installed = True


def _resolve_base_url(provider: str) -> str:
    default_urls = {
        "anthropic": "https://api.anthropic.com",
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    if base_url:
        return base_url
    return default_urls.get(provider, "")


def _next_trace_seq() -> int:
    global _trace_seq
    with _trace_lock:
        _trace_seq += 1
        return _trace_seq


def _trace_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_trace_text(text: str, limit: int = 50_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _jsonify_trace_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, dict):
        return {str(k): _jsonify_trace_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_trace_value(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonify_trace_value(value.model_dump(mode="json"))
        except TypeError:
            return _jsonify_trace_value(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonify_trace_value(value.dict())
    if hasattr(value, "__dict__"):
        data = {}
        for key, item in vars(value).items():
            if key.startswith("_") or callable(item):
                continue
            data[key] = _jsonify_trace_value(item)
        return data
    return str(value)


def _response_to_trace_body(response: Any) -> str:
    try:
        if hasattr(response, "model_dump_json"):
            try:
                return _truncate_trace_text(response.model_dump_json())
            except TypeError:
                return _truncate_trace_text(response.model_dump_json(indent=None))
        if hasattr(response, "json"):
            return _truncate_trace_text(response.json())
        return _truncate_trace_text(json.dumps(_jsonify_trace_value(response)))
    except Exception:
        fallback = {
            "error": "trace_response_serialization_failed",
            "type": type(response).__name__,
            "value": str(response),
        }
        return json.dumps(fallback)


def _error_to_trace_body(exc: Exception) -> str:
    payload = {
        "error": str(exc),
        "type": type(exc).__name__,
    }
    body = getattr(exc, "body", None)
    if body not in (None, ""):
        payload["body"] = _jsonify_trace_value(body)
    return _truncate_trace_text(json.dumps(payload))


def _http_status_from_error(exc: Exception) -> int:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else 0


def _trace_url_for_agent(agent, base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    api_mode = getattr(agent, "api_mode", "")

    if api_mode == "anthropic_messages":
        if base.endswith("/v1/messages"):
            return base
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    if api_mode == "codex_responses":
        if base.endswith("/responses"):
            return base
        return f"{base}/responses"

    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _append_trace_entry(
    *,
    url: str,
    request: dict[str, Any],
    latency_ms: float,
    http_status: int,
    raw_response: str,
) -> None:
    if not TRACE_FILE:
        return

    entry = {
        "seq": _next_trace_seq(),
        "timestamp": _trace_timestamp(),
        "type": "llm_call",
        "url": url,
        "latency_ms": round(latency_ms),
        "http_status": http_status,
        "request": request,
        "raw_response": raw_response,
    }
    try:
        with open(TRACE_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"[hermes-worker] Warning: could not write trace: {e}")


def _install_trace_logging(agent, base_url: str, *, thinking: str = "") -> None:
    if not TRACE_FILE or getattr(agent, "_gaia2_trace_logging_installed", False):
        return

    def _wrap_api_call(fn):
        @wraps(fn)
        def _wrapped(api_kwargs, *args, **kwargs):
            depth = getattr(_trace_depth, "value", 0)
            _trace_depth.value = depth + 1
            should_log = depth == 0
            request = {}
            if should_log:
                logged_api_kwargs = _normalize_chat_reasoning_request(
                    api_kwargs,
                    api_mode=getattr(agent, "api_mode", ""),
                    thinking=thinking,
                )
                request = _jsonify_trace_value(logged_api_kwargs)
            started = time.monotonic() if should_log else 0.0

            try:
                response = fn(api_kwargs, *args, **kwargs)
            except Exception as exc:
                if should_log:
                    _append_trace_entry(
                        url=_trace_url_for_agent(agent, base_url),
                        request=request,
                        latency_ms=(time.monotonic() - started) * 1000,
                        http_status=_http_status_from_error(exc),
                        raw_response=_error_to_trace_body(exc),
                    )
                raise
            else:
                if should_log:
                    _append_trace_entry(
                        url=_trace_url_for_agent(agent, base_url),
                        request=request,
                        latency_ms=(time.monotonic() - started) * 1000,
                        http_status=200,
                        raw_response=_response_to_trace_body(response),
                    )
                return response
            finally:
                _trace_depth.value = depth

        return _wrapped

    agent._interruptible_api_call = _wrap_api_call(agent._interruptible_api_call)
    agent._interruptible_streaming_api_call = _wrap_api_call(
        agent._interruptible_streaming_api_call
    )
    agent._gaia2_trace_logging_installed = True


def _create_agent():
    """Create and return a Hermes AIAgent instance."""
    _patch_hermes_time()
    from run_agent import AIAgent

    _patch_hermes_tool_call_ids(AIAgent)

    provider = os.environ.get("PROVIDER", "anthropic").strip()
    api_key = os.environ.get("API_KEY", "")
    model = os.environ.get("MODEL", "claude-opus-4-6")
    base_url = _resolve_base_url(provider)

    if not base_url:
        raise ValueError(
            f"Unsupported Hermes provider {provider!r}. Set BASE_URL explicitly."
        )

    # Read the rendered system prompt (scenario-specific tool list)
    try:
        with open(AGENTS_MD) as f:
            system_prompt = f.read()
        print(f"[hermes-worker] Loaded system prompt from {AGENTS_MD}")
    except OSError as e:
        print(f"[hermes-worker] Warning: could not read {AGENTS_MD}: {e}")
        system_prompt = ""

    kwargs = {
        "api_key": api_key,
        "model": model,
        "enabled_toolsets": ["terminal"],
        "max_iterations": 90,
        "quiet_mode": True,
        "tool_delay": 0.0,
        "skip_context_files": True,
        "skip_memory": True,
        "ephemeral_system_prompt": system_prompt,
    }
    if base_url:
        kwargs["base_url"] = base_url

    if provider:
        kwargs["provider"] = provider

    thinking = os.environ.get("THINKING", "")
    if thinking:
        kwargs["reasoning_config"] = {"effort": thinking}

    max_tokens = os.environ.get("MAX_TOKENS", "")
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)

    print(
        f"[hermes-worker] Creating AIAgent: model={model} base_url={base_url} provider={provider or 'auto'}"
    )
    agent = AIAgent(**kwargs)
    _disable_hermes_streaming_for_gaia2(agent)
    _patch_chat_reasoning_effort(agent, thinking)
    _install_trace_logging(agent, base_url, thinking=thinking)
    print("[hermes-worker] AIAgent created successfully")

    return agent


# ═══════════════════════════════════════════════════════════════════════
#  Socket communication
# ═══════════════════════════════════════════════════════════════════════

_sock_lock = threading.Lock()


def _send(sock: socket.socket, msg: dict) -> None:
    """Send a JSON line to the socket."""
    with _sock_lock:
        sock.sendall((json.dumps(msg) + "\n").encode())


def _recv_lines(sock: socket.socket):
    """Yield JSON-parsed lines from the socket."""
    buf = b""
    while True:
        data = sock.recv(65536)
        if not data:
            return
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                yield json.loads(line)


def _connect_to_adapter() -> socket.socket:
    """Connect to the adapter's Unix socket, retrying until it is ready."""
    deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS
    last_error: OSError | None = None

    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            print(f"[hermes-worker] Connecting to {WORKER_SOCK}")
            sock.connect(WORKER_SOCK)
            print("[hermes-worker] Connected")
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(CONNECT_RETRY_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Timed out connecting to Hermes adapter socket {WORKER_SOCK}: {last_error}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════


def main():
    # Disable Hermes's built-in PII redaction
    os.environ["HERMES_REDACT_SECRETS"] = "0"

    agent = _create_agent()
    conversation_history = []

    # Connect to adapter's Unix socket
    sock = _connect_to_adapter()

    _send(sock, {"type": "ready"})

    # Queue for incoming messages. A listener thread reads from the socket
    # and puts messages here. This allows interrupt messages to be delivered
    # to the agent while run_conversation() is blocking the main thread.
    import queue

    msg_queue = queue.Queue()

    def _listener():
        """Read messages from socket and dispatch interrupts or queue them."""
        for msg in _recv_lines(sock):
            msg_type = msg.get("type")
            if msg_type == "interrupt":
                text = msg.get("text", "")
                print(f"[hermes-worker] Interrupting: {text[:80]!r}")
                agent.interrupt(text)
            else:
                msg_queue.put(msg)

    listener = threading.Thread(target=_listener, daemon=True)
    listener.start()

    while True:
        msg = msg_queue.get()
        msg_type = msg.get("type")

        if msg_type != "message":
            print(f"[hermes-worker] Unknown message type: {msg_type}")
            continue

        text = msg.get("text", "")
        run_id = msg.get("run_id", "unknown")

        try:
            print(
                f"[hermes-worker] Starting conversation (run_id={run_id}): {text[:100]!r}"
            )
            agent.clear_interrupt()

            result = agent.run_conversation(
                user_message=text,
                conversation_history=list(conversation_history),
            )

            final_response = result.get("final_response", "") or ""
            messages = result.get("messages", [])

            conversation_history.clear()
            conversation_history.extend(messages)

            print(
                f"[hermes-worker] Conversation complete (run_id={run_id}): {final_response[:100]!r}"
            )
            _send(
                sock,
                {
                    "type": "response",
                    "run_id": run_id,
                    "state": "final",
                    "message": final_response,
                },
            )

        except Exception as e:
            print(f"[hermes-worker] Conversation error (run_id={run_id}): {e}")
            _send(
                sock,
                {
                    "type": "response",
                    "run_id": run_id,
                    "state": "error",
                    "message": f"Error: {e}",
                    "errorMessage": str(e),
                },
            )


if __name__ == "__main__":
    main()
