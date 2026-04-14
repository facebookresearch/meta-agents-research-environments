# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for gaia2_runner.trace_parser — trace.jsonl parsing and normalization.

Covers SSE reassembly (Anthropic/OpenAI), JSON response parsing (Anthropic/
OpenAI/Llama native), entry normalization, JSONL loading, and provider detection.
"""

from __future__ import annotations

import json
from pathlib import Path

from gaia2_runner.trace_parser import (
    canonicalize_entry,
    detect_format,
    load_jsonl,
    load_llm_calls,
    normalize_entry,
    parse_json_response,
    parse_jsonl_text,
    parse_sse_events,
    reassemble_anthropic,
    reassemble_openai,
)

# ---------------------------------------------------------------------------
# parse_sse_events
# ---------------------------------------------------------------------------


class TestParseSseEvents:
    def test_basic_parsing(self) -> None:
        text = 'data: {"type": "message_start"}\ndata: {"type": "message_stop"}\n'
        events = parse_sse_events(text)
        assert len(events) == 2
        assert events[0]["type"] == "message_start"

    def test_ignores_done_sentinel(self) -> None:
        text = 'data: {"type": "test"}\ndata: [DONE]\n'
        events = parse_sse_events(text)
        assert len(events) == 1

    def test_ignores_non_data_lines(self) -> None:
        text = (
            "event: message_start\n"
            'data: {"type": "message_start"}\n'
            "\n"
            "id: 123\n"
            ": comment\n"
        )
        events = parse_sse_events(text)
        assert len(events) == 1

    def test_skips_malformed_json(self) -> None:
        text = 'data: NOT_JSON\ndata: {"valid": true}\n'
        events = parse_sse_events(text)
        assert len(events) == 1
        assert events[0]["valid"] is True

    def test_empty_input(self) -> None:
        assert parse_sse_events("") == []


# ---------------------------------------------------------------------------
# reassemble_anthropic
# ---------------------------------------------------------------------------


class TestReassembleAnthropic:
    def test_text_response(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {
                    "model": "claude-opus-4-6",
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello "},
            },
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "world"},
            },
            {"type": "content_block_stop"},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 25},
            },
        ]

        result = reassemble_anthropic(events)

        assert result["model"] == "claude-opus-4-6"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 25
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello world"

    def test_tool_use_response(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {"model": "claude-opus-4-6", "usage": {"input_tokens": 50}},
            },
            {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "t1", "name": "exec"},
            },
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"command":'},
            },
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '"ls -la"}'},
            },
            {"type": "content_block_stop"},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 10},
            },
        ]

        result = reassemble_anthropic(events)

        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["name"] == "exec"
        assert block["input"] == {"command": "ls -la"}

    def test_thinking_block(self) -> None:
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 50}}},
            {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            },
            {"type": "content_block_stop"},
        ]

        result = reassemble_anthropic(events)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "Let me think..."

    def test_cache_tokens(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 100,
                        "cache_read_input_tokens": 500,
                        "cache_creation_input_tokens": 200,
                    },
                },
            },
        ]

        result = reassemble_anthropic(events)
        assert result["usage"]["cache_read_input_tokens"] == 500
        assert result["usage"]["cache_creation_input_tokens"] == 200

    def test_empty_events(self) -> None:
        result = reassemble_anthropic([])
        assert result["content"] == []
        assert result["usage"] == {}


# ---------------------------------------------------------------------------
# reassemble_openai
# ---------------------------------------------------------------------------


class TestReassembleOpenai:
    def test_text_response(self) -> None:
        events = [
            {"model": "gpt-4", "choices": [{"delta": {"content": "Hello "}}]},
            {"model": "gpt-4", "choices": [{"delta": {"content": "world"}}]},
            {"model": "gpt-4", "choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"usage": {"prompt_tokens": 50, "completion_tokens": 10}},
        ]

        result = reassemble_openai(events)

        assert result["model"] == "gpt-4"
        assert result["content"] == "Hello world"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 50

    def test_tool_calls(self) -> None:
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "exec", "arguments": '{"com'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'mand":"ls"}'},
                                }
                            ]
                        }
                    }
                ]
            },
        ]

        result = reassemble_openai(events)
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "exec"
        assert tc["function"]["arguments"] == '{"command":"ls"}'

    def test_reasoning_content(self) -> None:
        events = [
            {
                "model": "fireworks-kimi-k2p5-od",
                "choices": [{"delta": {"reasoning_content": "Plan "}}],
            },
            {
                "model": "fireworks-kimi-k2p5-od",
                "choices": [{"delta": {"reasoning_content": "the steps"}}],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            },
        ]

        result = reassemble_openai(events)

        assert result["model"] == "fireworks-kimi-k2p5-od"
        assert result["reasoning"] == "Plan the steps"
        assert result["finish_reason"] == "tool_calls"
        assert result["usage"]["prompt_tokens"] == 50

    def test_empty_events(self) -> None:
        result = reassemble_openai([])
        assert result["usage"] == {}
        assert result["finish_reason"] == ""


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_anthropic_format(self) -> None:
        data = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "model": "claude-opus-4-6",
        }
        result = parse_json_response(json.dumps(data))

        assert result["model"] == "claude-opus-4-6"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1

    def test_openai_format(self) -> None:
        data = {
            "choices": [
                {
                    "message": {"content": "Hello", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            "model": "gpt-4",
        }
        result = parse_json_response(json.dumps(data))

        assert result["model"] == "gpt-4"
        assert result["content"] == "Hello"
        assert result["finish_reason"] == "stop"

    def test_openai_format_reasoning_content(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "role": "assistant",
                        "reasoning_content": "Check the calendar first.",
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            "model": "fireworks-kimi-k2p5-od",
        }
        result = parse_json_response(json.dumps(data))

        assert result["model"] == "fireworks-kimi-k2p5-od"
        assert result["reasoning"] == "Check the calendar first."
        assert result["finish_reason"] == "tool_calls"

    def test_provider_native_format(self) -> None:
        data = {
            "completion_message": {
                "content": {"type": "text", "text": "Hello from provider"},
                "stop_reason": "end_of_turn",
            },
            "metrics": [
                {"metric": "num_prompt_tokens", "value": 100},
                {"metric": "num_completion_tokens", "value": 50},
            ],
        }
        result = parse_json_response(json.dumps(data))

        assert result["content"] == "Hello from provider"
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 50

    def test_provider_native_with_tool_calls(self) -> None:
        data = {
            "completion_message": {
                "content": "",
                "tool_calls": [{"id": "t1", "function": {"name": "exec"}}],
            },
        }
        result = parse_json_response(json.dumps(data))
        assert result["tool_calls"] == [{"id": "t1", "function": {"name": "exec"}}]

    def test_invalid_json(self) -> None:
        result = parse_json_response("NOT JSON")
        assert result == {}

    def test_empty_string(self) -> None:
        result = parse_json_response("")
        assert result == {}

    def test_google_generate_content_format(self) -> None:
        data = {
            "modelVersion": "gemini-3.1-pro",
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Consider options", "thought": True},
                            {
                                "functionCall": {
                                    "name": "exec",
                                    "args": {"command": "date"},
                                }
                            },
                            {"text": "Done"},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 10,
                "totalTokenCount": 60,
            },
        }

        result = parse_json_response(json.dumps(data))

        assert result["model"] == "gemini-3.1-pro"
        assert result["content"] == "Done"
        assert result["reasoning"] == "Consider options"
        assert result["finish_reason"] == "stop"
        assert result["tool_calls"] == [
            {
                "id": "google-call-2",
                "type": "function",
                "function": {
                    "name": "exec",
                    "arguments": '{"command":"date"}',
                },
            }
        ]
        assert result["usage"]["prompt_tokens"] == 50
        assert result["usage"]["completion_tokens"] == 10


# ---------------------------------------------------------------------------
# normalize_entry
# ---------------------------------------------------------------------------


class TestNormalizeEntry:
    def test_format_b_passthrough(self) -> None:
        """Entries with structured 'response' dict are returned as-is."""
        entry = {
            "type": "llm_call",
            "response": {"content": [{"type": "text", "text": "Hello"}]},
        }
        result = normalize_entry(entry)
        assert result is entry  # same object

    def test_format_a_anthropic_sse(self) -> None:
        """Raw SSE response is parsed into structured response."""
        sse = (
            'data: {"type":"message_start","message":{"model":"claude-opus-4-6",'
            '"usage":{"input_tokens":100}}}\n'
            'data: {"type":"content_block_start","content_block":{"type":"text","text":""}}\n'
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n'
            'data: {"type":"content_block_stop"}\n'
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":10}}\n'
        )
        entry = {"type": "llm_call", "raw_response": sse, "http_status": 200}

        result = normalize_entry(entry)

        assert "response" in result
        assert result["response"]["model"] == "claude-opus-4-6"
        assert result["response"]["status"] == 200
        assert len(result["response"]["content"]) == 1

    def test_format_a_json_response(self) -> None:
        """Plain JSON raw_response is parsed."""
        data = {
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 10},
            "model": "claude-opus-4-6",
        }
        entry = {"type": "llm_call", "raw_response": json.dumps(data)}

        result = normalize_entry(entry)

        assert result["response"]["model"] == "claude-opus-4-6"

    def test_format_a_openai_responses_sse_with_tool_call(self) -> None:
        sse = (
            'data: {"type":"response.created","response":{"model":"gpt-5.4","usage":null}}\n'
            'data: {"type":"response.reasoning_summary_part.added","item_id":"rs_1","part":{"type":"summary_text","text":""}}\n'
            'data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":"Inspect contacts"}\n'
            'data: {"type":"response.completed","response":{"object":"response","status":"completed","model":"gpt-5.4","usage":{"input_tokens":12,"output_tokens":7},"output":[{"id":"rs_1","type":"reasoning","summary":[]},{"id":"fc_1","type":"function_call","call_id":"call_1","name":"exec","arguments":"{\\"command\\":\\"contacts --help\\"}"}]}}\n'
        )
        entry = {"type": "llm_call", "raw_response": sse, "http_status": 200}

        result = normalize_entry(entry)

        assert result["response"]["model"] == "gpt-5.4"
        assert result["response"]["status"] == 200
        assert result["response"]["finish_reason"] == "tool_calls"
        assert result["response"]["reasoning"] == "Inspect contacts"
        assert result["response"]["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "exec",
                    "arguments": '{"command":"contacts --help"}',
                },
            }
        ]

    def test_format_a_openai_responses_sse_with_message(self) -> None:
        sse = (
            'data: {"type":"response.created","response":{"model":"gpt-5.4","usage":null}}\n'
            'data: {"type":"response.completed","response":{"object":"response","status":"completed","model":"gpt-5.4","usage":{"input_tokens":20,"output_tokens":10},"output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Done"}]}]}}\n'
        )
        entry = {"type": "llm_call", "raw_response": sse, "http_status": 200}

        result = normalize_entry(entry)

        assert result["response"]["model"] == "gpt-5.4"
        assert result["response"]["content"] == "Done"
        assert result["response"]["finish_reason"] == "stop"

    def test_format_a_openai_responses_sse_truncated_fallback(self) -> None:
        sse = (
            'data: {"type":"response.created","response":{"model":"gpt-5.4","usage":null}}\n'
            'data: {"type":"response.reasoning_summary_part.added","item_id":"rs_1","part":{"type":"summary_text","text":""}}\n'
            'data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":"Plan tool call"}\n'
            'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"exec"}}\n'
            'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"command\\":\\"ls"}\n'
            'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\\"command\\":\\"ls\\"}"}\n'
            'data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\\"command\\":\\"ls\\"}","call_id":"call_1","name":"exec"}}\n'
        )
        entry = {"type": "llm_call", "raw_response": sse, "http_status": 200}

        result = normalize_entry(entry)

        assert result["response"]["model"] == "gpt-5.4"
        assert result["response"]["reasoning"] == "Plan tool call"
        assert result["response"]["finish_reason"] == "tool_calls"
        assert result["response"]["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "exec", "arguments": '{"command":"ls"}'},
            }
        ]

    def test_format_a_google_sse_with_tool_call(self) -> None:
        sse = (
            'data: {"modelVersion":"gemini-3.1-pro","candidates":[{"content":{"parts":[{"text":"Plan","thought":true}]}}]}\n'
            'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"exec","args":{"command":"date"}}}]}}]}\n'
            'data: {"candidates":[{"content":{"parts":[{"text":"Done"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":14}}\n'
        )
        entry = {"type": "llm_call", "raw_response": sse, "http_status": 200}

        result = normalize_entry(entry)

        assert result["response"]["model"] == "gemini-3.1-pro"
        assert result["response"]["reasoning"] == "Plan"
        assert result["response"]["content"] == "Done"
        assert result["response"]["finish_reason"] == "stop"
        assert result["response"]["tool_calls"] == [
            {
                "id": "google-call-1",
                "type": "function",
                "function": {"name": "exec", "arguments": '{"command":"date"}'},
            }
        ]
        assert result["response"]["usage"]["prompt_tokens"] == 10

    def test_empty_raw_response(self) -> None:
        entry = {"type": "llm_call", "raw_response": ""}
        result = normalize_entry(entry)
        assert "response" not in result or result.get("raw_response") == ""

    def test_does_not_mutate_original(self) -> None:
        entry = {
            "type": "llm_call",
            "raw_response": '{"content": [], "stop_reason": "end_turn"}',
        }
        original_keys = set(entry.keys())
        normalize_entry(entry)
        assert set(entry.keys()) == original_keys


class TestCanonicalizeEntry:
    def test_anthropic_thinking_blocks_are_canonicalized(self) -> None:
        entry = {
            "type": "llm_call",
            "request": {"messages": [{"role": "user", "content": "Do the task"}]},
            "response": {
                "model": "claude-opus-4-6",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [
                    {"type": "thinking", "thinking": "Plan the steps."},
                    {
                        "type": "tool_use",
                        "name": "exec",
                        "input": {"command": "date"},
                    },
                ],
                "stop_reason": "tool_use",
            },
        }

        result = canonicalize_entry(entry)

        assert result["response_blocks"] == [
            {"type": "thinking", "thinking": "Plan the steps."},
            {"type": "tool_use", "name": "exec", "input": {"command": "date"}},
        ]

    def test_google_request_and_response_are_canonicalized(self) -> None:
        entry = {
            "type": "llm_call",
            "request": {
                "systemInstruction": {"parts": [{"text": "System prompt"}]},
                "contents": [
                    {"role": "user", "parts": [{"text": "Do the task"}]},
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "exec",
                                    "args": {"command": "date"},
                                }
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "exec",
                                    "response": {"output": "Thu Jan 1"},
                                }
                            }
                        ],
                    },
                ],
            },
            "raw_response": json.dumps(
                {
                    "modelVersion": "gemini-3.1-pro",
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "Answer"}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                }
            ),
            "http_status": 200,
        }

        result = canonicalize_entry(entry)

        assert result["system_prompt"] == "System prompt"
        assert [msg["role"] for msg in result["request_messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert result["request_messages"][1]["content_blocks"] == [
            {"type": "text", "text": "Do the task"}
        ]
        assert result["request_messages"][2]["content_blocks"] == [
            {"type": "tool_use", "name": "exec", "input": {"command": "date"}}
        ]
        assert result["request_messages"][3]["content_blocks"] == [
            {"type": "tool_result", "content": '{"output":"Thu Jan 1"}', "name": "exec"}
        ]
        assert result["response_blocks"] == [{"type": "text", "text": "Answer"}]
        assert result["response_stop_reason"] == "stop"


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_anthropic_url(self) -> None:
        entry = {"url": "https://api.anthropic.com/v1/messages"}
        assert detect_format(entry) == "anthropic"

    def test_openai_url(self) -> None:
        entry = {"url": "https://api.openai.com/v1/chat/completions"}
        assert detect_format(entry) == "openai"

    def test_chat_completions_url(self) -> None:
        entry = {"url": "https://provider.example/chat/completions"}
        assert detect_format(entry) == "openai"

    def test_no_url(self) -> None:
        entry = {}
        assert detect_format(entry) == "openai"  # default


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


class TestParseJsonlText:
    def test_basic(self) -> None:
        text = '{"a": 1}\n{"b": 2}\n'
        entries = parse_jsonl_text(text)
        assert len(entries) == 2

    def test_skips_blank_lines(self) -> None:
        text = '{"a": 1}\n\n{"b": 2}\n\n'
        entries = parse_jsonl_text(text)
        assert len(entries) == 2

    def test_skips_malformed(self) -> None:
        text = '{"a": 1}\nNOT JSON\n{"b": 2}\n'
        entries = parse_jsonl_text(text)
        assert len(entries) == 2

    def test_empty(self) -> None:
        assert parse_jsonl_text("") == []


class TestLoadJsonl:
    def test_loads_file(self, tmp_path: Path) -> None:
        p = tmp_path / "data.jsonl"
        p.write_text('{"x": 1}\n{"y": 2}\n')
        entries = load_jsonl(p)
        assert len(entries) == 2

    def test_missing_file(self, tmp_path: Path) -> None:
        entries = load_jsonl(tmp_path / "nonexistent.jsonl")
        assert entries == []


class TestLoadLlmCalls:
    def test_filters_llm_calls(self, tmp_path: Path) -> None:
        p = tmp_path / "trace.jsonl"
        lines = [
            json.dumps({"type": "llm_call", "seq": 1}),
            json.dumps({"type": "other", "seq": 2}),
            json.dumps({"type": "llm_call", "seq": 3}),
        ]
        p.write_text("\n".join(lines) + "\n")

        calls = load_llm_calls(p)
        assert len(calls) == 2
        assert calls[0]["seq"] == 1
        assert calls[1]["seq"] == 3

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "trace.jsonl"
        p.write_text("")
        assert load_llm_calls(p) == []
