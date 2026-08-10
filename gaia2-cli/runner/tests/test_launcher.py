# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from gaia2_runner.launcher import (
    ApptainerLauncher,
    LocalLauncher,
    _write_private_file,
)


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


def _apptainer_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[ApptainerLauncher, Path, dict[str, list[str]]]:
    """An ApptainerLauncher with scratch redirected under tmp_path."""
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"not-a-real-sif")
    scratch_root = tmp_path / "scratch"
    captured: dict[str, list[str]] = {}

    monkeypatch.delenv("GAIA2_OC_SIF_STAGE_LOCAL", raising=False)
    monkeypatch.delenv("GAIA2_PROXY_RELAY_URL", raising=False)
    monkeypatch.delenv("GAIA2_CA_BUNDLE", raising=False)
    monkeypatch.setattr(
        ApptainerLauncher,
        "_instance_scratch_dir",
        staticmethod(lambda container_id: scratch_root / container_id),
    )

    launcher = ApptainerLauncher(image_sif=str(sif))

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    class _FakePopen:
        def __init__(self, args: list[str], **kwargs: object) -> None:
            captured["popen"] = args

    monkeypatch.setattr(launcher, "_run", fake_run)
    monkeypatch.setattr("gaia2_runner.launcher.subprocess.Popen", _FakePopen)
    return launcher, scratch_root, captured


def test_apptainer_env_file_is_owner_only_and_not_on_command_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher, scratch_root, captured = _apptainer_launcher(monkeypatch, tmp_path)

    container_id = launcher.launch(
        "localhost/gaia2-oc:latest",
        str(scenario_path),
        provider="anthropic",
        model="some-model",
        api_key="sk-super-secret",
    )

    env_file = scratch_root / container_id / "secrets" / "env.sh"
    assert env_file.is_file()
    # The credentials live in the env file...
    assert "sk-super-secret" in env_file.read_text()
    # ...which must be readable by nobody but the launching UID.
    assert env_file.stat().st_mode & 0o777 == 0o600
    assert env_file.parent.stat().st_mode & 0o777 == 0o700
    # ...and must never leak onto an argv visible in `ps`.
    assert not any("sk-super-secret" in arg for arg in captured["args"])
    assert not any("sk-super-secret" in arg for arg in captured["popen"])
    assert f"{env_file}:/var/gaia2/env.sh:ro" in captured["args"]


def test_write_private_file_tightens_preexisting_loose_perms(tmp_path: Path) -> None:
    stale = tmp_path / "secrets" / "env.sh"
    stale.parent.mkdir(parents=True)
    stale.write_text("export API_KEY=old\n")
    stale.chmod(0o644)
    stale.parent.chmod(0o755)

    _write_private_file(stale, "export API_KEY=new\n")

    assert stale.stat().st_mode & 0o777 == 0o600
    assert stale.parent.stat().st_mode & 0o777 == 0o700
    assert stale.read_text() == "export API_KEY=new\n"


def test_write_private_file_ignores_permissive_umask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_umask = os.umask(0o000)
    try:
        _write_private_file(tmp_path / "d" / "env.sh", "export API_KEY=k\n")
    finally:
        os.umask(old_umask)

    assert (tmp_path / "d" / "env.sh").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "d").stat().st_mode & 0o777 == 0o700


def test_apptainer_stop_removes_env_file_even_when_scratch_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}")
    launcher, scratch_root, _ = _apptainer_launcher(monkeypatch, tmp_path)
    monkeypatch.setenv("GAIA2_OC_KEEP_SCRATCH", "1")

    container_id = launcher.launch(
        "localhost/gaia2-oc:latest",
        str(scenario_path),
        provider="anthropic",
        api_key="sk-super-secret",
    )
    env_file = scratch_root / container_id / "secrets" / "env.sh"
    assert env_file.is_file()

    launcher.stop(container_id)

    assert not env_file.exists()
    # Scratch itself is kept for debugging.
    assert (scratch_root / container_id).is_dir()


def test_build_provider_env_skips_agent_env_for_oracle() -> None:
    pairs = LocalLauncher._build_provider_env(
        "localhost/gaia2-oracle:latest",
        provider=None,
        model=None,
        api_key=None,
        env={},
    )

    assert pairs == []
