# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from gaia2_runner.launcher import LocalLauncher


def _env_map(args: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for idx, arg in enumerate(args):
        if arg == "-e":
            key, value = args[idx + 1].split("=", 1)
            env[key] = value
    return env


def test_local_launcher_adds_extra_volumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher = LocalLauncher(runtime="podman")
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.setattr("gaia2_runner.launcher.os.path.isfile", lambda _: False)
    monkeypatch.setattr(launcher, "_run", fake_run)

    container_id = launcher.launch(
        "localhost/gaia2-oc:latest",
        str(scenario_path),
        extra_volumes=(
            "/tmp/traces:/tmp/traces",
            "/tmp/extra:/tmp/extra",
        ),
    )

    assert container_id == "container-123"
    assert "/tmp/traces:/tmp/traces" in captured["args"]
    assert "/tmp/extra:/tmp/extra" in captured["args"]


def test_local_launcher_publishes_adapter_port_for_podman_on_macos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher = LocalLauncher(runtime="podman")
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.setattr("gaia2_runner.launcher.sys.platform", "darwin")
    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.setattr("gaia2_runner.launcher.os.path.isfile", lambda _: False)
    monkeypatch.setattr(launcher, "_run", fake_run)

    launcher.launch(
        "localhost/gaia2-oracle:latest",
        str(scenario_path),
        adapter_port=8123,
        gateway_port=18790,
    )

    args = captured["args"]
    env = _env_map(args)
    assert "--network=host" not in args
    assert "-p" in args
    assert "127.0.0.1:8123:8123" in args
    assert env["GAIA2_ADAPTER_PORT"] == "8123"
    assert env["OPENCLAW_GATEWAY_PORT"] == "18790"
    assert env["OPENCLAW_GATEWAY_URL"] == "ws://127.0.0.1:18790"


def test_local_launcher_keeps_host_network_for_podman_on_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher = LocalLauncher(runtime="podman")
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.setattr("gaia2_runner.launcher.sys.platform", "linux")
    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.setattr("gaia2_runner.launcher.os.path.isfile", lambda _: False)
    monkeypatch.setattr(launcher, "_run", fake_run)

    launcher.launch(
        "localhost/gaia2-oracle:latest",
        str(scenario_path),
        adapter_port=8123,
    )

    args = captured["args"]
    assert "--network=host" in args
    assert "-p" not in args


def test_local_launcher_keeps_explicit_network_for_podman_on_macos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher = LocalLauncher(runtime="podman")
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.setattr("gaia2_runner.launcher.sys.platform", "darwin")
    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.setattr("gaia2_runner.launcher.os.path.isfile", lambda _: False)
    monkeypatch.setattr(launcher, "_run", fake_run)

    launcher.launch(
        "localhost/gaia2-oracle:latest",
        str(scenario_path),
        network="bridge",
        adapter_port=8123,
    )

    args = captured["args"]
    assert "--network=bridge" in args
    assert "-p" not in args


def test_local_launcher_uses_configured_proxy_relay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher = LocalLauncher(runtime="podman")
    captured: dict[str, list[str]] = {}
    relay: dict[str, tuple[str, int]] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    def fake_ensure_proxy_relay(host: str, port: int) -> None:
        relay["target"] = (host, port)

    monkeypatch.setenv("GAIA2_PROXY_RELAY_URL", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", ".corp.example")
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "gaia2_runner.launcher._ensure_proxy_relay", fake_ensure_proxy_relay
    )
    monkeypatch.setattr("gaia2_runner.launcher.os.path.isfile", lambda _: False)
    monkeypatch.setattr(launcher, "_run", fake_run)

    launcher.launch("localhost/gaia2-oc:latest", str(scenario_path))

    env = _env_map(captured["args"])
    assert relay["target"] == ("proxy.example", 8080)
    assert env["http_proxy"] == "http://127.0.0.1:18888"
    assert env["https_proxy"] == "http://127.0.0.1:18888"
    assert env["HTTP_PROXY"] == "http://127.0.0.1:18888"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:18888"
    assert "127.0.0.1" in env["NO_PROXY"].split(",")
    assert "localhost" in env["NO_PROXY"].split(",")
    assert ".corp.example" in env["NO_PROXY"].split(",")
    assert env["no_proxy"] == env["NO_PROXY"]


def test_local_launcher_mounts_configured_ca_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("dummy-ca\n")
    launcher = LocalLauncher(runtime="podman")
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.setenv("GAIA2_CA_BUNDLE", str(ca_bundle))
    monkeypatch.setattr(launcher, "_run", fake_run)

    launcher.launch("localhost/gaia2-oc:latest", str(scenario_path))

    env = _env_map(captured["args"])
    assert f"{ca_bundle}:/etc/ssl/certs/gaia2-host-ca-bundle.crt:ro" in captured["args"]
    assert env["NODE_EXTRA_CA_CERTS"] == "/etc/ssl/certs/gaia2-host-ca-bundle.crt"
    assert env["REQUESTS_CA_BUNDLE"] == "/etc/ssl/certs/gaia2-host-ca-bundle.crt"
    assert env["SSL_CERT_FILE"] == "/etc/ssl/certs/gaia2-host-ca-bundle.crt"


def test_build_provider_env_adds_google_secondary_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "google-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    pairs = LocalLauncher._build_provider_env(
        "localhost/gaia2-oc:latest",
        provider="google",
        model="custom-google-model",
        api_key=None,
        env={},
    )

    assert ("PROVIDER", "google") in pairs
    assert ("MODEL", "custom-google-model") in pairs
    assert ("GEMINI_API_KEY", "google-key") in pairs
    assert ("GOOGLE_API_KEY", "google-key") in pairs


def test_build_provider_env_uses_openai_api_key_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("API_KEY", raising=False)

    with caplog.at_level("WARNING"):
        pairs = LocalLauncher._build_provider_env(
            "localhost/gaia2-oc:latest",
            provider="openai",
            model="test-openai-model",
            api_key=None,
            env={},
        )

    assert ("API_KEY", "openai-key") in pairs
    assert ("OPENAI_API_KEY", "openai-key") in pairs
    assert "pulling from OPENAI_API_KEY" in caplog.text


def test_build_provider_env_skips_agent_env_for_oracle() -> None:
    pairs = LocalLauncher._build_provider_env(
        "localhost/gaia2-oracle:latest",
        provider=None,
        model=None,
        api_key=None,
        env={},
    )

    assert pairs == []


def test_local_launcher_seeds_dirs_before_starting_the_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher = LocalLauncher(runtime="podman")
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.setattr("gaia2_runner.launcher.os.path.isfile", lambda _: False)
    monkeypatch.setattr(launcher, "_run", fake_run)

    container_id = launcher.launch(
        "localhost/gaia2-oc:latest",
        str(scenario_path),
        seed_dirs={
            "/home/agent": "/tmp/carried/home",
            "/var/gaia2/carried_state": "/tmp/carried/app_state",
        },
    )

    assert container_id == "container-123"
    assert calls[0][:2] == ["podman", "create"]
    assert "-d" not in calls[0]
    assert calls[1:] == [
        ["podman", "cp", "/tmp/carried/home/.", "container-123:/home/agent"],
        [
            "podman",
            "cp",
            "/tmp/carried/app_state/.",
            "container-123:/var/gaia2/carried_state",
        ],
        ["podman", "start", "container-123"],
    ]


def test_local_launcher_removes_the_container_when_seeding_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher = LocalLauncher(runtime="podman")
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1] == "cp":
            raise subprocess.CalledProcessError(125, args)
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.setattr("gaia2_runner.launcher.os.path.isfile", lambda _: False)
    monkeypatch.setattr(launcher, "_run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        launcher.launch(
            "localhost/gaia2-oc:latest",
            str(scenario_path),
            seed_dirs={"/home/agent": "/tmp/carried/home"},
        )

    assert ["podman", "rm", "-f", "container-123"] in calls
    assert not any(args[1] == "start" for args in calls)


def test_local_launcher_stages_the_judge_context_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario_1.json"
    scenario_path.write_text("{}")
    launcher = LocalLauncher(runtime="podman")
    captured: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.setattr(launcher, "_run", fake_run)

    # Without a judge context the command is exactly what it always was.
    launcher.launch("localhost/gaia2-oc:latest", str(scenario_path))
    without = captured[-1]
    assert not any("judge_context" in arg for arg in without)

    # With one, it is mounted read-only under /var/gaia2 (closed to the agent)
    # for the init entrypoint to install as the daemon's private copy.
    judge_context = tmp_path / "scenario_1.judge_context.json"
    judge_context.write_text('{"version": 1, "turns": {}}')
    launcher.launch("localhost/gaia2-oc:latest", str(scenario_path))
    with_context = captured[-1]
    mount = f"{judge_context}:/var/gaia2/judge_context.staged.json:ro"
    idx = with_context.index(mount)
    assert with_context[idx - 1] == "-v"
    # Nothing else changes (container names carry a random suffix).
    rest = with_context[: idx - 1] + with_context[idx + 1 :]
    assert [a for a in rest if not a.startswith("--name=")] == [
        a for a in without if not a.startswith("--name=")
    ]
