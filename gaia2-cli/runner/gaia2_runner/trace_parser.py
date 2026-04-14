# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared trace.jsonl parsing for Gaia2 LLM trace data.

Parses raw trace entries (with raw_response strings) into normalized
dicts with structured response data. Handles:
- SSE streaming responses (Anthropic, OpenAI Chat Completions, Responses API)
- Plain JSON responses (Anthropic, OpenAI, OpenAI Responses, provider-native)
- JSONL loading and filtering
- Provider/model detection

Used by both the trace viewer and the SFT exporter within gaia2_runner.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── SSE reassembly ────────────────────────────────────────────────────────


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    """Parse SSE ``data:`` lines into a list of JSON objects.

    Handles both Anthropic and OpenAI streaming formats.  Ignores
    ``[DONE]`` sentinel and non-JSON lines.
    """
    events: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return events


def reassemble_anthropic(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble Anthropic SSE events into a structured response.

    Handles the ``message_start`` → ``content_block_start`` →
    ``content_block_delta`` → ``message_delta`` event sequence.
    """
    content_blocks: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    stop_reason: str = ""
    model: str = ""
    current_block: dict[str, Any] | None = None

    for ev in events:
        ev_type = ev.get("type", "")

        if ev_type == "message_start":
            msg = ev.get("message", {})
            model = msg.get("model", "")
            u = msg.get("usage", {})
            usage["input_tokens"] = u.get("input_tokens", 0)
            usage["cache_creation_input_tokens"] = u.get(
                "cache_creation_input_tokens", 0
            )
            usage["cache_read_input_tokens"] = u.get("cache_read_input_tokens", 0)

        elif ev_type == "content_block_start":
            cb = ev.get("content_block", {})
            current_block = dict(cb)
            # Initialize text accumulator for text blocks
            if current_block.get("type") == "text":
                current_block["text"] = current_block.get("text", "")
            elif current_block.get("type") == "thinking":
                current_block["thinking"] = current_block.get("thinking", "")

        elif ev_type == "content_block_delta":
            delta = ev.get("delta", {})
            if current_block is not None:
                if delta.get("type") == "text_delta":
                    current_block["text"] = current_block.get("text", "") + delta.get(
                        "text", ""
                    )
                elif delta.get("type") == "thinking_delta":
                    current_block["thinking"] = current_block.get(
                        "thinking", ""
                    ) + delta.get("thinking", "")
                elif delta.get("type") == "input_json_delta":
                    current_block.setdefault("_partial_json", "")
                    current_block["_partial_json"] += delta.get("partial_json", "")

        elif ev_type == "content_block_stop":
            if current_block is not None:
                # Finalize tool_use input from accumulated JSON
                if (
                    current_block.get("type") == "tool_use"
                    and "_partial_json" in current_block
                ):
                    try:
                        current_block["input"] = json.loads(
                            current_block.pop("_partial_json")
                        )
                    except json.JSONDecodeError:
                        current_block.pop("_partial_json", None)
                content_blocks.append(current_block)
                current_block = None

        elif ev_type == "message_delta":
            delta = ev.get("delta", {})
            stop_reason = delta.get("stop_reason", stop_reason)
            u = ev.get("usage", {})
            if u.get("output_tokens"):
                usage["output_tokens"] = u["output_tokens"]

    result: dict[str, Any] = {
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": usage,
    }
    if model:
        result["model"] = model
    return result


def reassemble_openai(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble OpenAI SSE events into a structured response.

    Accumulates ``choices[].delta`` text and tool_calls into a single
    response dict.
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str = ""
    usage: dict[str, Any] = {}
    model: str = ""

    for ev in events:
        model = model or ev.get("model", "")
        if ev.get("usage"):
            usage = ev["usage"]

        for choice in ev.get("choices", []):
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

            # Text content
            if delta.get("content"):
                text_parts.append(delta["content"])

            reasoning = _extract_openai_chat_reasoning(delta)
            if not reasoning:
                reasoning = _extract_openai_chat_reasoning(choice.get("message", {}))
            if reasoning:
                reasoning_parts.append(reasoning)

            # Tool calls
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.get("id"):
                    tool_calls[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_calls[idx]["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_calls[idx]["function"]["arguments"] += fn["arguments"]

    result: dict[str, Any] = {"usage": usage}
    if model:
        result["model"] = model

    content = "".join(text_parts)
    if content:
        result["content"] = content

    reasoning = "".join(reasoning_parts)
    if reasoning:
        result["reasoning"] = reasoning

    if tool_calls:
        result["tool_calls"] = [tool_calls[k] for k in sorted(tool_calls)]

    result["finish_reason"] = finish_reason
    return result


def _json_dumps_compact(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        return str(value)


def _coerce_openai_reasoning_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part
            for part in (_coerce_openai_reasoning_text(item) for item in value)
            if part
        )
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        if value.get("summary") not in (None, ""):
            return _coerce_openai_reasoning_text(value.get("summary"))
        return _json_dumps_compact(value)
    return str(value)


def _extract_openai_chat_reasoning(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    reasoning = _coerce_openai_reasoning_text(payload.get("reasoning_content"))
    if reasoning:
        return reasoning
    return _coerce_openai_reasoning_text(payload.get("reasoning"))


def _extract_responses_message_text(item: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for part in item.get("content", []):
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")
        if part_type in {"output_text", "text"} and part.get("text"):
            text_parts.append(part["text"])
        elif part_type == "refusal" and part.get("refusal"):
            text_parts.append(part["refusal"])
    return "".join(text_parts)


def _extract_responses_reasoning_text(item: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for part in item.get("summary", []):
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")
        if part_type in {"summary_text", "text"} and part.get("text"):
            text_parts.append(part["text"])
    return "".join(text_parts)


def _normalize_openai_responses_response(response: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"usage": response.get("usage") or {}}
    model = response.get("model", "")
    if model:
        result["model"] = model
    if response.get("error") is not None:
        result["error"] = response["error"]

    message_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        if item_type == "message":
            text = _extract_responses_message_text(item)
            if text:
                message_parts.append(text)
        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "") or "",
                    },
                }
            )
        elif item_type == "reasoning":
            text = _extract_responses_reasoning_text(item)
            if text:
                reasoning_parts.append(text)

    content = "".join(message_parts)
    if content:
        result["content"] = content
    if tool_calls:
        result["tool_calls"] = tool_calls
    reasoning = "".join(reasoning_parts)
    if reasoning:
        result["reasoning"] = reasoning

    result["finish_reason"] = (
        "tool_calls" if tool_calls else ("stop" if content else "")
    )
    return result


def _normalize_google_finish_reason(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    upper = value.upper()
    if upper == "STOP":
        return "stop"
    if upper == "MAX_TOKENS":
        return "length"
    return value.lower()


def _google_usage_to_usage(data: dict[str, Any]) -> dict[str, Any]:
    usage_meta = data.get("usageMetadata", {})
    if not isinstance(usage_meta, dict):
        return {}

    usage: dict[str, Any] = {}
    if isinstance(usage_meta.get("promptTokenCount"), int):
        usage["prompt_tokens"] = usage_meta["promptTokenCount"]
    if isinstance(usage_meta.get("candidatesTokenCount"), int):
        usage["completion_tokens"] = usage_meta["candidatesTokenCount"]
    if isinstance(usage_meta.get("totalTokenCount"), int):
        usage["total_tokens"] = usage_meta["totalTokenCount"]
    return usage


def _normalize_google_function_call(
    function_call: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    return {
        "id": function_call.get("id")
        or function_call.get("callId")
        or f"google-call-{index + 1}",
        "type": "function",
        "function": {
            "name": function_call.get("name", ""),
            "arguments": _json_dumps_compact(function_call.get("args", {})),
        },
    }


def _append_google_tool_call(
    tool_calls: list[dict[str, Any]],
    tool_call: dict[str, Any],
) -> None:
    if tool_calls:
        last = tool_calls[-1]
        last_name = last.get("function", {}).get("name", "")
        new_name = tool_call.get("function", {}).get("name", "")
        last_args = last.get("function", {}).get("arguments", "")
        new_args = tool_call.get("function", {}).get("arguments", "")

        # Native Gemini streaming can resend the same function call with
        # progressively fuller args; keep the newest version instead of
        # duplicating it in the viewer.
        if (
            last_name
            and last_name == new_name
            and (
                not last_args
                or not new_args
                or str(last_args).startswith(str(new_args))
                or str(new_args).startswith(str(last_args))
            )
        ):
            merged = dict(last)
            merged["function"] = dict(last.get("function", {}))
            merged["function"]["arguments"] = new_args or last_args
            if not merged.get("id"):
                merged["id"] = tool_call.get("id", "")
            tool_calls[-1] = merged
            return

    if tool_call not in tool_calls:
        tool_calls.append(tool_call)


def _normalize_google_generate_content_response(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"usage": _google_usage_to_usage(data)}

    model = data.get("modelVersion") or data.get("model", "")
    if model:
        result["model"] = model

    if data.get("error") is not None:
        result["error"] = data["error"]

    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        if data.get("promptFeedback") is not None:
            result["prompt_feedback"] = data["promptFeedback"]
        return result

    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    finish_reason = _normalize_google_finish_reason(candidate.get("finishReason"))
    if finish_reason:
        result["finish_reason"] = finish_reason

    message_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    content = candidate.get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue

        if isinstance(part.get("functionCall"), dict):
            _append_google_tool_call(
                tool_calls,
                _normalize_google_function_call(part["functionCall"], index=idx),
            )
            continue

        text = part.get("text")
        if isinstance(text, str) and text:
            if part.get("thought"):
                reasoning_parts.append(text)
            else:
                message_parts.append(text)

    content_text = "".join(message_parts)
    if content_text:
        result["content"] = content_text

    if tool_calls:
        result["tool_calls"] = tool_calls

    reasoning_text = "".join(reasoning_parts)
    if reasoning_text:
        result["reasoning"] = reasoning_text

    return result


def reassemble_google_generative_ai(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble native Google GenerateContent SSE events."""

    result: dict[str, Any] = {"usage": {}}
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for event in events:
        parsed = _normalize_google_generate_content_response(event)

        if parsed.get("model") and not result.get("model"):
            result["model"] = parsed["model"]

        if parsed.get("usage"):
            result["usage"] = parsed["usage"]

        if parsed.get("error") is not None:
            result["error"] = parsed["error"]

        if parsed.get("prompt_feedback") is not None:
            result["prompt_feedback"] = parsed["prompt_feedback"]

        if parsed.get("finish_reason"):
            result["finish_reason"] = parsed["finish_reason"]

        if parsed.get("content"):
            content_parts.append(str(parsed["content"]))

        if parsed.get("reasoning"):
            reasoning_parts.append(str(parsed["reasoning"]))

        for tool_call in parsed.get("tool_calls", []):
            _append_google_tool_call(tool_calls, tool_call)

    if content_parts:
        result["content"] = "".join(content_parts)
    if reasoning_parts:
        result["reasoning"] = "".join(reasoning_parts)
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def reassemble_openai_responses(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble OpenAI Responses SSE events into a structured response.

    Prefer the final ``response.completed`` payload when present because it
    contains the full normalized output. When traces are truncated to 50KB,
    fall back to incremental ``response.*`` events so tool calls and final
    assistant text still render in the viewer.
    """

    model: str = ""
    usage: dict[str, Any] = {}
    error: Any = None
    completed_response: dict[str, Any] | None = None

    message_parts_by_item: dict[str, str] = {}
    reasoning_parts_by_item: dict[str, str] = {}
    tool_calls_by_item: dict[str, dict[str, Any]] = {}

    for ev in events:
        ev_type = ev.get("type", "")

        if ev_type in {"response.created", "response.in_progress"}:
            response = ev.get("response", {})
            model = model or response.get("model", "")
            if response.get("usage"):
                usage = response["usage"]
            if response.get("error") is not None:
                error = response["error"]

        elif (
            ev_type == "response.output_item.added"
            or ev_type == "response.output_item.done"
        ):
            item = ev.get("item", {})
            if not isinstance(item, dict):
                continue
            item_id = item.get("id", "")
            item_type = item.get("type", "")
            if item_type == "function_call":
                tool = tool_calls_by_item.setdefault(
                    item_id,
                    {
                        "id": item.get("call_id") or item.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if item.get("call_id") or item.get("id"):
                    tool["id"] = item.get("call_id") or item.get("id", "")
                if item.get("name"):
                    tool["function"]["name"] = item["name"]
                if item.get("arguments"):
                    tool["function"]["arguments"] = item["arguments"]
            elif item_type == "message":
                text = _extract_responses_message_text(item)
                if text:
                    message_parts_by_item.setdefault(item_id, text)
            elif item_type == "reasoning":
                text = _extract_responses_reasoning_text(item)
                if text:
                    reasoning_parts_by_item[item_id] = text

        elif ev_type == "response.function_call_arguments.delta":
            item_id = ev.get("item_id", "")
            tool = tool_calls_by_item.setdefault(
                item_id,
                {
                    "id": item_id,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            tool["function"]["arguments"] += ev.get("delta", "")

        elif ev_type == "response.function_call_arguments.done":
            item_id = ev.get("item_id", "")
            tool = tool_calls_by_item.setdefault(
                item_id,
                {
                    "id": item_id,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if ev.get("arguments"):
                tool["function"]["arguments"] = ev["arguments"]

        elif ev_type == "response.output_text.delta":
            item_id = ev.get("item_id", "")
            message_parts_by_item.setdefault(item_id, "")
            message_parts_by_item[item_id] += ev.get("delta", "")

        elif ev_type == "response.output_text.done":
            item_id = ev.get("item_id", "")
            if not message_parts_by_item.get(item_id) and ev.get("text"):
                message_parts_by_item[item_id] = ev["text"]

        elif ev_type == "response.reasoning_summary_part.added":
            item_id = ev.get("item_id", "")
            part = ev.get("part", {})
            if isinstance(part, dict) and part.get("text"):
                reasoning_parts_by_item.setdefault(item_id, "")
                reasoning_parts_by_item[item_id] += part["text"]
            else:
                reasoning_parts_by_item.setdefault(item_id, "")

        elif ev_type == "response.reasoning_summary_text.delta":
            item_id = ev.get("item_id", "")
            reasoning_parts_by_item.setdefault(item_id, "")
            reasoning_parts_by_item[item_id] += ev.get("delta", "")

        elif ev_type == "response.completed":
            completed_response = ev.get("response", {})
            model = model or completed_response.get("model", "")
            if completed_response.get("usage"):
                usage = completed_response["usage"]
            if completed_response.get("error") is not None:
                error = completed_response["error"]

    if completed_response:
        result = _normalize_openai_responses_response(completed_response)
        if not result.get("reasoning"):
            reasoning = "".join(reasoning_parts_by_item.values())
            if reasoning:
                result["reasoning"] = reasoning
        return result

    result = {"usage": usage}
    if model:
        result["model"] = model
    if error is not None:
        result["error"] = error

    content = "".join(message_parts_by_item.values())
    if content:
        result["content"] = content

    tool_calls = list(tool_calls_by_item.values())
    if tool_calls:
        result["tool_calls"] = tool_calls

    reasoning = "".join(reasoning_parts_by_item.values())
    if reasoning:
        result["reasoning"] = reasoning

    result["finish_reason"] = (
        "tool_calls" if tool_calls else ("stop" if content else "")
    )
    return result


# ── JSON response parsing ─────────────────────────────────────────────────


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a plain JSON (non-streaming) response body.

    Handles Anthropic Messages API, OpenAI Chat Completions, OpenAI
    Responses API, and Llama native response formats.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}

    # Anthropic format: has "content" list and "stop_reason"
    if "content" in data and "stop_reason" in data:
        return {
            "content": data.get("content", []),
            "stop_reason": data.get("stop_reason", ""),
            "usage": data.get("usage", {}),
            "model": data.get("model", ""),
        }

    # Provider-native format: has "completion_message"
    if "completion_message" in data:
        msg = data["completion_message"]
        result: dict[str, Any] = {
            "usage": {},
            "model": data.get("model", ""),
            "finish_reason": msg.get("stop_reason", ""),
        }
        content = msg.get("content")
        if content:
            if isinstance(content, dict) and content.get("type") == "text":
                result["content"] = content.get("text", "")
            else:
                result["content"] = content
        if msg.get("tool_calls"):
            result["tool_calls"] = msg["tool_calls"]
        # Extract metrics into usage
        for m in data.get("metrics", []):
            if m.get("metric") == "num_prompt_tokens":
                result["usage"]["prompt_tokens"] = m["value"]
            elif m.get("metric") == "num_completion_tokens":
                result["usage"]["completion_tokens"] = m["value"]
        return result

    # OpenAI Responses format: has "object":"response" and "output"
    if data.get("object") == "response" and isinstance(data.get("output"), list):
        return _normalize_openai_responses_response(data)

    # Native Google GenerateContent format: has "candidates" and usageMetadata.
    if (
        isinstance(data.get("candidates"), list)
        or data.get("promptFeedback") is not None
    ):
        return _normalize_google_generate_content_response(data)

    # OpenAI format: has "choices"
    if "choices" in data:
        choice = data["choices"][0] if data["choices"] else {}
        msg = choice.get("message", {})
        result: dict[str, Any] = {
            "usage": data.get("usage", {}),
            "model": data.get("model", ""),
            "finish_reason": choice.get("finish_reason", ""),
        }
        if msg.get("content"):
            result["content"] = msg["content"]
        if msg.get("tool_calls"):
            result["tool_calls"] = msg["tool_calls"]
        reasoning = _extract_openai_chat_reasoning(msg)
        if reasoning:
            result["reasoning"] = reasoning
        return result

    return data


def _content_blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif block_type == "thinking" and block.get("thinking"):
            parts.append(str(block["thinking"]))
    return "".join(parts)


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return _json_dumps_compact(content)


def _normalize_content_blocks(content: Any) -> list[dict[str, Any]]:
    if content in (None, "", []):
        return []

    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            blocks.extend(_normalize_content_blocks(item))
        return blocks

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, dict):
        return [{"type": "text", "text": str(content)}]

    if isinstance(content.get("functionCall"), dict):
        function_call = content["functionCall"]
        return [
            {
                "type": "tool_use",
                "name": function_call.get("name", ""),
                "input": function_call.get("args", {}),
            }
        ]

    if isinstance(content.get("functionResponse"), dict):
        function_response = content["functionResponse"]
        return [
            {
                "type": "tool_result",
                "content": _tool_result_text(function_response.get("response", {})),
                "name": function_response.get("name", ""),
            }
        ]

    block_type = content.get("type", "")
    thinking = content.get("thinking")
    if block_type == "thinking" and isinstance(thinking, str):
        return [{"type": "thinking", "thinking": thinking}]

    text = content.get("text")
    if isinstance(text, str):
        if content.get("thought"):
            return [{"type": "thinking", "thinking": text}]
        if block_type in {"thinking"}:
            return [{"type": "thinking", "thinking": text}]
        if block_type in {"text", "input_text", "output_text", ""}:
            return [{"type": "text", "text": text}]

    if block_type == "tool_use":
        return [
            {
                "type": "tool_use",
                "name": content.get("name", ""),
                "input": content.get("input", {}),
            }
        ]
    if block_type == "tool_result":
        return [
            {
                "type": "tool_result",
                "content": _tool_result_text(content.get("content", "")),
            }
        ]

    raw_type = block_type or "raw"
    return [
        {"type": "raw", "raw_type": raw_type, "value": _json_dumps_compact(content)}
    ]


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(tool_calls):
        if not isinstance(item, dict):
            continue

        if isinstance(item.get("function"), dict):
            normalized.append(
                {
                    "id": item.get("id", "") or f"tool-call-{index + 1}",
                    "type": item.get("type", "function") or "function",
                    "function": {
                        "name": item["function"].get("name", ""),
                        "arguments": _json_dumps_compact(
                            item["function"].get("arguments", "")
                        ),
                    },
                }
            )
            continue

        if item.get("type") == "function_call":
            normalized.append(
                {
                    "id": item.get("call_id")
                    or item.get("id")
                    or f"tool-call-{index + 1}",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": _json_dumps_compact(item.get("arguments", "")),
                    },
                }
            )

    return normalized


def normalize_request_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Return provider-neutral request history for viewer rendering."""

    messages: list[dict[str, Any]] = []

    system = request.get("system")
    if system:
        messages.append(
            {
                "role": "system",
                "content_blocks": _normalize_content_blocks(system),
                "tool_calls": [],
            }
        )

    system_instruction = request.get("systemInstruction")
    if isinstance(system_instruction, dict):
        system_parts = _normalize_content_blocks(system_instruction.get("parts", []))
        if system_parts:
            messages.append(
                {
                    "role": "system",
                    "content_blocks": system_parts,
                    "tool_calls": [],
                }
            )

    raw_messages = request.get("messages")
    if isinstance(raw_messages, list):
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content_blocks": _normalize_content_blocks(
                        item.get("content", "")
                    ),
                    "tool_calls": _normalize_tool_calls(item.get("tool_calls", [])),
                }
            )
        return messages

    input_items = request.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            if role:
                messages.append(
                    {
                        "role": "system" if role == "developer" else role,
                        "content_blocks": _normalize_content_blocks(
                            item.get("content", "")
                        ),
                        "tool_calls": _normalize_tool_calls(item.get("tool_calls", [])),
                    }
                )
                continue

            item_type = item.get("type", "")
            if item_type == "message":
                messages.append(
                    {
                        "role": item.get("role", "assistant"),
                        "content_blocks": _normalize_content_blocks(
                            item.get("content", "")
                        ),
                        "tool_calls": _normalize_tool_calls(item.get("tool_calls", [])),
                    }
                )
            elif item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "content_blocks": [
                            {
                                "type": "tool_result",
                                "content": _tool_result_text(item.get("output", "")),
                            }
                        ],
                        "tool_calls": [],
                    }
                )
            elif item_type == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content_blocks": [],
                        "tool_calls": _normalize_tool_calls([item]),
                    }
                )
        return messages

    contents = request.get("contents")
    if isinstance(contents, list):
        for item in contents:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "user")
            role = "assistant" if role == "model" else role
            messages.append(
                {
                    "role": role,
                    "content_blocks": _normalize_content_blocks(item.get("parts", [])),
                    "tool_calls": [],
                }
            )
        return messages

    return messages


def canonicalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Attach provider-neutral request/response fields for the trace viewer."""

    normalized = normalize_entry(entry)
    result = dict(normalized)

    request = result.get("request", {})
    if isinstance(request, dict):
        request_messages = normalize_request_messages(request)
    else:
        request_messages = []

    system_prompt = ""
    if request_messages and request_messages[0].get("role") == "system":
        system_prompt = _content_blocks_to_text(
            request_messages[0].get("content_blocks", [])
        )

    response = result.get("response", {})
    response_blocks = (
        _normalize_content_blocks(response.get("content", ""))
        if isinstance(response, dict)
        else []
    )
    response_tool_calls = (
        _normalize_tool_calls(response.get("tool_calls", []))
        if isinstance(response, dict)
        else []
    )
    response_reasoning = ""
    if isinstance(response, dict) and isinstance(response.get("reasoning"), str):
        response_reasoning = response["reasoning"]

    response_stop_reason = ""
    if isinstance(response, dict):
        response_stop_reason = str(
            response.get("stop_reason", "") or response.get("finish_reason", "")
        ).strip()

    response_status = 0
    if isinstance(response, dict):
        raw_status = response.get("status", result.get("http_status", 0))
        response_status = raw_status if isinstance(raw_status, int) else 0

    error_title = ""
    error_detail = ""
    if isinstance(response, dict):
        error_title = str(response.get("title", "") or "")
        error_detail = str(response.get("detail", "") or "")
        response_error = response.get("error")
        if isinstance(response_error, dict):
            error_title = error_title or str(response_error.get("type", "") or "")
            error_detail = error_detail or str(response_error.get("message", "") or "")
        elif isinstance(response_error, str):
            error_detail = error_detail or response_error

    result["request_messages"] = request_messages
    result["system_prompt"] = system_prompt
    result["response_blocks"] = response_blocks
    result["response_tool_calls"] = response_tool_calls
    result["response_reasoning"] = response_reasoning
    result["response_stop_reason"] = response_stop_reason
    result["response_status"] = response_status
    result["response_error"] = {"title": error_title, "detail": error_detail}
    return result


# ── Entry normalization ───────────────────────────────────────────────────


def normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize a trace entry so it always has a structured ``response`` dict.

    - Format B (has ``response`` dict) → returned as-is.
    - Format A (has ``raw_response`` string) → parsed into a ``response``
      dict with structured ``content``, ``usage``, ``stop_reason``.

    The ``url`` and ``http_status`` fields from Format A are preserved
    for provider detection.
    """
    # Already normalized (Format B)
    if "response" in entry and isinstance(entry.get("response"), dict):
        return entry

    raw = entry.get("raw_response", "")
    if not raw:
        return entry

    # Copy entry so we don't mutate the original
    result = dict(entry)

    # Detect format: SSE vs plain JSON
    is_sse = "data: " in raw[:200]

    if is_sse:
        events = parse_sse_events(raw)
        if not events:
            result["response"] = {"content": [], "usage": {}}
            return result

        # Detect Anthropic vs OpenAI SSE by first event's structure
        first = events[0]
        if first.get("type") in (
            "message_start",
            "content_block_start",
            "content_block_delta",
            "ping",
        ):
            parsed = reassemble_anthropic(events)
        elif (
            isinstance(first.get("candidates"), list)
            or first.get("promptFeedback") is not None
        ):
            parsed = reassemble_google_generative_ai(events)
        elif str(first.get("type", "")).startswith("response."):
            parsed = reassemble_openai_responses(events)
        else:
            parsed = reassemble_openai(events)
    else:
        parsed = parse_json_response(raw)

    # Map http_status → response.status
    if "http_status" in entry:
        parsed["status"] = entry["http_status"]

    result["response"] = parsed
    return result


# ── JSONL loading ─────────────────────────────────────────────────────────


def parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    """Parse JSONL text into list of dicts."""
    entries: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning empty list on failure."""
    entries: list[dict[str, Any]] = []
    try:
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    except Exception:
        pass
    return entries


def load_llm_calls(trace_path: Path) -> list[dict[str, Any]]:
    """Load all llm_call entries from a trace.jsonl file."""
    entries = []
    for line in trace_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("type") == "llm_call":
                entries.append(entry)
        except json.JSONDecodeError:
            continue
    return entries


# ── Provider / model detection ────────────────────────────────────────────


def detect_format(entry: dict[str, Any]) -> str:
    """Detect whether a trace entry is Anthropic or OpenAI format.

    Returns "anthropic" or "openai".
    """
    url = entry.get("url", "")
    if "/v1/messages" in url:
        return "anthropic"
    return "openai"


def extract_model(trace_entries: list[dict[str, Any]], result: dict[str, Any]) -> str:
    """Extract model name from trace entries or result."""
    for entry in trace_entries:
        model = entry.get("request", {}).get("model", "")
        if model:
            return model
        model = entry.get("response", {}).get("model", "")
        if model:
            return model
    return ""


def extract_provider(
    trace_entries: list[dict[str, Any]], result: dict[str, Any]
) -> str:
    """Infer provider from trace entries.

    Checks the ``url`` field first (Format A), then falls back to
    response structure heuristics.
    """
    for entry in trace_entries:
        url = entry.get("url", "")
        if url:
            if "anthropic" in url or "/messages" in url:
                return "anthropic"
            if "generateContent" in url or "streamGenerateContent" in url:
                return "google"
            if "google" in url and "/chat/completions" not in url:
                return "google"
            if "/responses" in url or "/chat/completions" in url:
                return "openai-compat"

        resp = entry.get("response", {})
        if resp.get("stop_reason"):
            if "content" in resp and isinstance(resp["content"], list):
                return "anthropic"
        model = resp.get("model", "")
        if isinstance(model, str) and model.startswith("gemini"):
            return "google"
        if resp.get("tool_calls"):
            return "openai-compat"
    return ""
