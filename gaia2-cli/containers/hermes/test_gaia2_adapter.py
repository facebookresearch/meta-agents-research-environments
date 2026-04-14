# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for Hermes's Gaia2 adapter."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest import mock

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SHARED_DIR = os.path.join(_REPO_ROOT, "shared")
for _path in (
    _THIS_DIR,
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "cli"),
    os.path.join(_REPO_ROOT, "core"),
    _SHARED_DIR,
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gaia2_adapter as adapter  # noqa: E402


@pytest.fixture(autouse=True)
def reset_adapter_state(tmp_path, monkeypatch: pytest.MonkeyPatch):
    aui_mock = mock.MagicMock()
    buffer_mock = mock.MagicMock(side_effect=lambda response: {"seq": 1, **response})

    monkeypatch.setattr(adapter, "WORKER_SOCK", str(tmp_path / "hermes-worker.sock"))
    monkeypatch.setattr(adapter, "WORKER_READY_TIMEOUT", 2.0)
    monkeypatch.setattr(adapter, "write_aui_event", aui_mock)
    monkeypatch.setattr(adapter._state, "buffer_and_broadcast", buffer_mock)

    adapter._worker_server = None
    adapter._worker_reader = None
    adapter._worker_writer = None
    adapter._worker_ready = False
    adapter._worker_ready_event = None
    adapter._worker_send_lock = None
    adapter._pending_run_ids.clear()

    yield {
        "aui": aui_mock,
        "buffer": buffer_mock,
    }

    asyncio.run(adapter._shutdown_worker_bridge())


class TestBackendResponse:
    def test_final_response_buffers_and_writes_turn_boundary(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        response = {
            "state": "final",
            "run_id": "run-1",
            "message": "Here are the results.",
        }
        adapter._pending_run_ids.append("run-1")

        adapter.on_backend_response(response)

        reset_adapter_state["buffer"].assert_called_once_with(response)
        reset_adapter_state["aui"].assert_called_once_with(
            "send_message_to_user",
            "Here are the results.",
        )
        assert list(adapter._pending_run_ids) == []

    def test_error_response_writes_turn_boundary(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        response = {
            "state": "error",
            "run_id": "run-2",
            "message": "Error: request failed.",
        }

        adapter.on_backend_response(response)

        reset_adapter_state["buffer"].assert_called_once_with(response)
        reset_adapter_state["aui"].assert_called_once_with(
            "send_message_to_user",
            "Error: request failed.",
        )


class TestWorkerBridge:
    def test_backend_connect_and_send_message(self) -> None:
        async def _exercise() -> None:
            connect_task = asyncio.create_task(adapter.backend_connect())

            for _ in range(50):
                if os.path.exists(adapter.WORKER_SOCK):
                    break
                await asyncio.sleep(0.01)

            reader, writer = await asyncio.open_unix_connection(adapter.WORKER_SOCK)
            writer.write(b'{"type":"ready"}\n')
            await writer.drain()
            await connect_task

            result = await adapter.send_message("hello")
            line = await asyncio.wait_for(reader.readline(), timeout=1)
            frame = json.loads(line)

            assert frame["type"] == "message"
            assert frame["text"] == "hello"
            assert frame["run_id"] == result["run_id"]
            assert adapter.is_connected() is True

            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.05)

        asyncio.run(_exercise())

    def test_second_message_interrupts_active_run(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        async def _exercise() -> None:
            connect_task = asyncio.create_task(adapter.backend_connect())

            for _ in range(50):
                if os.path.exists(adapter.WORKER_SOCK):
                    break
                await asyncio.sleep(0.01)

            reader, writer = await asyncio.open_unix_connection(adapter.WORKER_SOCK)
            writer.write(b'{"type":"ready"}\n')
            await writer.drain()
            await connect_task

            first = await adapter.send_message("first task")
            second = await adapter.send_message("second task")

            msg1 = json.loads(await asyncio.wait_for(reader.readline(), timeout=1))
            msg2 = json.loads(await asyncio.wait_for(reader.readline(), timeout=1))
            msg3 = json.loads(await asyncio.wait_for(reader.readline(), timeout=1))

            assert msg1 == {
                "type": "message",
                "text": "first task",
                "run_id": first["run_id"],
            }
            assert msg2 == {"type": "interrupt", "text": "second task"}
            assert msg3 == {
                "type": "message",
                "text": "second task",
                "run_id": second["run_id"],
            }

            writer.write(
                (
                    json.dumps(
                        {
                            "type": "response",
                            "run_id": first["run_id"],
                            "state": "error",
                            "message": "interrupted",
                        }
                    )
                    + "\n"
                ).encode()
            )
            writer.write(
                (
                    json.dumps(
                        {
                            "type": "response",
                            "run_id": second["run_id"],
                            "state": "final",
                            "message": "done",
                        }
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()
            await asyncio.sleep(0.05)

            assert reset_adapter_state["buffer"].call_count == 2
            assert list(adapter._pending_run_ids) == []

            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.05)

        asyncio.run(_exercise())
