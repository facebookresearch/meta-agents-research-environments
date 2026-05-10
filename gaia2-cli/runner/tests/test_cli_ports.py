# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import pytest
from gaia2_runner import cli as runner_cli


def test_allocate_ports_returns_distinct_adapter_and_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = iter([41000, 41000, 41001])

    monkeypatch.setattr(runner_cli, "_allocate_free_port", lambda: next(candidates))

    assert runner_cli._allocate_ports() == (41000, 41001)


def test_allocate_ports_skips_ports_reserved_by_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = iter([41000, 41000, 41001, 41002])
    reserved_ports = {41000}

    monkeypatch.setattr(runner_cli, "_allocate_free_port", lambda: next(candidates))

    assert runner_cli._allocate_ports(reserved_ports) == (41001, 41002)
    assert reserved_ports == {41000, 41001, 41002}


def test_allocate_ports_raises_after_repeated_reserved_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_cli, "PORT_ALLOCATION_ATTEMPTS", 3)
    monkeypatch.setattr(runner_cli, "_allocate_free_port", lambda: 41000)

    with pytest.raises(RuntimeError, match="unique free port"):
        runner_cli._allocate_ports({41000})
