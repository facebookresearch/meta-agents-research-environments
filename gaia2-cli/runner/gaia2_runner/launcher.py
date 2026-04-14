# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Container launcher for Gaia2 eval containers.

Provides ContainerLauncher ABC and LocalLauncher (podman/docker).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping
from urllib.parse import urlparse

logger: logging.Logger = logging.getLogger(__name__)

_RED = "\033[31m"
_RESET = "\033[0m"

# Default port for the local proxy relay (host process → upstream proxy).
_RELAY_PORT = 18888
_PROXY_RELAY_ENV = "GAIA2_PROXY_RELAY_URL"
_CA_BUNDLE_ENV = "GAIA2_CA_BUNDLE"
_CONTAINER_CA_BUNDLE = "/etc/ssl/certs/gaia2-host-ca-bundle.crt"
_HOST_CONTROL_ENV_KEYS = frozenset({_PROXY_RELAY_ENV, _CA_BUNDLE_ENV})

# Module-level proxy relay singleton — shared across all launcher instances
# so multiple concurrent containers reuse one relay.
_relay_lock = threading.Lock()
_relay_started = False
_relay_target: tuple[str, int] | None = None


def _allocate_free_port() -> int:
    """Allocate an ephemeral port by binding to port 0, then releasing it.

    There is a small TOCTOU window between releasing the port and the
    container binding it, but in practice this is fine for dev/eval use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Proxy relay ─────────────────────────────────────────────────────────


def _parse_proxy_relay_target(value: str) -> tuple[str, int]:
    """Parse ``GAIA2_PROXY_RELAY_URL`` into ``(host, port)``."""
    parsed = urlparse(value if "://" in value else f"http://{value}")
    if not parsed.hostname:
        raise ValueError(
            f"Invalid {_PROXY_RELAY_ENV} value {value!r}: missing proxy host"
        )
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, port


def _resolve_proxy_relay_target(
    env: Mapping[str, str] | None = None,
) -> tuple[str, int] | None:
    """Return the configured upstream proxy relay target, if any."""
    value = (
        (env or {}).get(_PROXY_RELAY_ENV) or os.environ.get(_PROXY_RELAY_ENV, "")
    ).strip()
    if not value:
        return None
    return _parse_proxy_relay_target(value)


def _resolve_ca_bundle_path(env: Mapping[str, str] | None = None) -> str | None:
    """Return the optional host CA bundle path, if any."""
    value = (
        (env or {}).get(_CA_BUNDLE_ENV) or os.environ.get(_CA_BUNDLE_ENV, "")
    ).strip()
    return value or None


def _merge_no_proxy(*values: str | None) -> str:
    """Merge comma-separated NO_PROXY values while preserving order."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for item in value.split(","):
            candidate = item.strip()
            if candidate and candidate not in seen:
                merged.append(candidate)
                seen.add(candidate)
    return ",".join(merged)


def _ensure_proxy_relay(proxy_host: str, proxy_port: int) -> None:
    """Start the proxy relay exactly once (idempotent singleton).

    Safe to call from multiple threads — only the first call starts the
    relay; subsequent calls return immediately.
    """
    global _relay_started, _relay_target
    with _relay_lock:
        target = (proxy_host, proxy_port)
        if _relay_started:
            if _relay_target != target:
                raise RuntimeError(
                    f"{_PROXY_RELAY_ENV} changed from {_relay_target} to {target} "
                    "within the same process"
                )
            return
        _start_proxy_relay(proxy_host, proxy_port)
        _relay_target = target
        _relay_started = True


def _start_proxy_relay(
    proxy_host: str,
    proxy_port: int,
    listen_port: int = _RELAY_PORT,
) -> threading.Thread:
    """Start a TCP relay on 127.0.0.1:listen_port → proxy_host:proxy_port.

    The relay runs as a daemon thread. It is useful when an outbound proxy
    authorises the host process identity but not the container's identity.

    Returns the listener thread (for bookkeeping; it's a daemon so it dies
    with the process).
    """

    def _relay(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except Exception:
            pass
        finally:
            # Signal the other direction that we're done writing.
            try:
                dst.shutdown(socket.SHUT_WR)
            except Exception:
                pass

    def _handle(client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection((proxy_host, proxy_port))
            t1 = threading.Thread(target=_relay, args=(client, upstream), daemon=True)
            t2 = threading.Thread(target=_relay, args=(upstream, client), daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        except Exception:
            pass
        finally:
            # Handler owns socket lifecycle — close exactly once.
            for s in (client, upstream):
                if s is not None:
                    try:
                        s.close()
                    except Exception:
                        pass

    def _listener() -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", listen_port))
        except OSError:
            logger.info("Proxy relay port %d already in use, reusing", listen_port)
            return
        srv.listen(128)
        logger.info(
            "Proxy relay listening on 127.0.0.1:%d -> %s:%d",
            listen_port,
            proxy_host,
            proxy_port,
        )
        while True:
            try:
                client, _ = srv.accept()
                threading.Thread(target=_handle, args=(client,), daemon=True).start()
            except Exception:
                pass

    t = threading.Thread(target=_listener, daemon=True)
    t.start()
    time.sleep(0.2)
    return t


# ── Launcher ABC ────────────────────────────────────────────────────────


class ContainerLauncher(ABC):
    """Abstract base for container lifecycle management."""

    @abstractmethod
    def launch(
        self,
        image: str,
        scenario_json_path: str,
        *,
        env: dict[str, str] | None = None,
        network: str = "host",
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        extra_volumes: tuple[str, ...] | None = None,
        adapter_port: int | None = None,
        gateway_port: int | None = None,
    ) -> str:
        """Start a container with the given scenario, return container_id.

        When *adapter_port* or *gateway_port* are set, the corresponding
        env vars are injected so the container uses non-default ports.
        This allows multiple containers to run concurrently with
        ``--network=host``.
        """

    @abstractmethod
    def exec(
        self,
        container_id: str,
        command: list[str],
        *,
        user: str | None = None,
    ) -> str:
        """Run a command inside the container, return stdout."""

    @abstractmethod
    def copy_from(
        self,
        container_id: str,
        src_path: str,
        dst_path: str,
    ) -> None:
        """Copy a file from the container to the host.

        Works even on stopped containers (unlike exec).
        """

    @abstractmethod
    def stop(self, container_id: str) -> None:
        """Stop and remove the container."""

    def get_host_adapter_port(self) -> int | None:
        """Return the host-side port where the adapter is reachable.

        For local launchers (podman), returns ``None`` — the caller should
        use the adapter_port it already knows about.  For remote launchers
        (VMVM), returns the tunnel port on the host that maps to port 8090
        inside the VM.
        """
        return None

    @staticmethod
    def _container_name(scenario_json_path: str) -> str:
        """Derive a readable container name from the scenario filename."""
        import uuid

        scenario_name = os.path.splitext(os.path.basename(scenario_json_path))[0]
        suffix = uuid.uuid4().hex[:8]
        return (
            "gaia2-"
            + "".join(c if c.isalnum() or c in "_.-" else "_" for c in scenario_name)
            + f"-{suffix}"
        )

    @staticmethod
    def _build_provider_env(
        image: str,
        *,
        provider: str | None,
        model: str | None,
        api_key: str | None,
        env: dict[str, str] | None,
    ) -> list[tuple[str, str]]:
        """Build provider/model/key env var pairs for the container.

        Returns a list of ``(key, value)`` pairs suitable for passing as
        ``-e key=value`` to podman/docker.
        """
        from .container_env import (
            detect_profile,
            provider_api_key_export_keys,
            resolve_api_key_details,
        )

        profile = detect_profile(image)
        if not profile.requires_agent_llm:
            return []

        effective_provider = provider or profile.default_provider
        resolved_key = resolve_api_key_details(effective_provider, api_key, env)
        effective_key = resolved_key.value
        pairs: list[tuple[str, str]] = []

        pairs.append((profile.provider_key, effective_provider))
        if effective_key:
            pairs.append((profile.api_key_key, effective_key))
        if model:
            pairs.append((profile.model_key, model))
        for k, v in profile.extra_flags.items():
            pairs.append((k, v))

        for key in provider_api_key_export_keys(effective_provider):
            if effective_key:
                pairs.append((key, effective_key))

        logger.info(
            "Using provider=%s, model=%s",
            effective_provider,
            model or "(default)",
        )
        if resolved_key.from_env and resolved_key.source:
            logger.warning(
                "%sAgent API key not passed via --api-key; pulling from %s%s",
                _RED,
                resolved_key.source,
                _RESET,
            )
        return pairs

    def wait_for_adapter(
        self,
        container_id: str,
        port: int = 8090,
        timeout: int = 120,
        interval: float = 2.0,
    ) -> None:
        """Poll the gaia2-adapter /health endpoint until it reports connected.

        Raises TimeoutError if the adapter doesn't become ready within timeout.
        """
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            try:
                out = self.exec(
                    container_id,
                    [
                        "/usr/bin/curl",
                        "-sf",
                        "-m",
                        "2",
                        "--noproxy",
                        "127.0.0.1",
                        f"http://127.0.0.1:{port}/health",
                    ],
                )
                health = json.loads(out)
                if health.get("connected"):
                    logger.info("Adapter is ready (port %d)", port)
                    return
                last_error = f"adapter not connected: {health}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(interval)
        raise TimeoutError(
            f"Adapter on port {port} not ready after {timeout}s: {last_error}"
        )


# ── Podman implementation ───────────────────────────────────────────────


class LocalLauncher(ContainerLauncher):
    """Container launcher using podman (or docker) CLI."""

    def __init__(self, runtime: str = "podman") -> None:
        self.runtime = runtime
        # Support multi-word runtimes like "sudo podman"
        self._rt = runtime.split()

    def _run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        logger.debug("Running: %s", " ".join(args))
        return subprocess.run(
            args, capture_output=True, text=True, check=True, **kwargs
        )

    def launch(
        self,
        image: str,
        scenario_json_path: str,
        *,
        env: dict[str, str] | None = None,
        network: str = "host",
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        extra_volumes: tuple[str, ...] | None = None,
        adapter_port: int | None = None,
        gateway_port: int | None = None,
    ) -> str:
        container_name = self._container_name(scenario_json_path)

        cmd = [
            *self._rt,
            "run",
            "-d",
            f"--name={container_name}",
            f"--network={network}",
            # Workaround for crun seccomp cache permission errors on
            # some kernels (linkat EPERM in seccomp bpf setup).
            "--security-opt",
            "seccomp=unconfined",
            "-v",
            f"{scenario_json_path}:/var/gaia2/custom_scenario.json:ro",
            "--entrypoint",
            "bash",
        ]
        for volume in extra_volumes or ():
            cmd.extend(["-v", volume])

        # ── Dynamic port allocation ─────────────────────────────────────
        if adapter_port is not None:
            cmd.extend(["-e", f"GAIA2_ADAPTER_PORT={adapter_port}"])
        if gateway_port is not None:
            cmd.extend(
                [
                    "-e",
                    f"OPENCLAW_GATEWAY_PORT={gateway_port}",
                    "-e",
                    f"OPENCLAW_GATEWAY_URL=ws://127.0.0.1:{gateway_port}",
                ]
            )

        # ── Provider / model / key ──────────────────────────────────────
        for key, val in self._build_provider_env(
            image, provider=provider, model=model, api_key=api_key, env=env
        ):
            cmd.extend(["-e", f"{key}={val}"])

        # ── Proxy setup ─────────────────────────────────────────────────
        relay_target = _resolve_proxy_relay_target(env)
        if relay_target:
            relay_host, relay_port = relay_target
            _ensure_proxy_relay(relay_host, relay_port)
            relay_url = f"http://127.0.0.1:{_RELAY_PORT}"
            no_proxy_value = _merge_no_proxy(
                "127.0.0.1,localhost",
                (env or {}).get("NO_PROXY"),
                (env or {}).get("no_proxy"),
                os.environ.get("NO_PROXY"),
                os.environ.get("no_proxy"),
            )
            cmd.extend(
                [
                    "-e",
                    f"http_proxy={relay_url}",
                    "-e",
                    f"https_proxy={relay_url}",
                    "-e",
                    f"HTTP_PROXY={relay_url}",
                    "-e",
                    f"HTTPS_PROXY={relay_url}",
                    "-e",
                    f"NO_PROXY={no_proxy_value}",
                    "-e",
                    f"no_proxy={no_proxy_value}",
                ]
            )
        else:
            for proxy_var in (
                "http_proxy",
                "https_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "no_proxy",
                "NO_PROXY",
            ):
                val = (env or {}).get(proxy_var) or os.environ.get(proxy_var, "")
                if val:
                    cmd.extend(["-e", f"{proxy_var}={val}"])

        # ── Host CA certs ───────────────────────────────────────────────
        host_ca = _resolve_ca_bundle_path(env)
        if host_ca and os.path.isfile(host_ca):
            cmd.extend(
                [
                    "-v",
                    f"{host_ca}:{_CONTAINER_CA_BUNDLE}:ro",
                    "-e",
                    f"NODE_EXTRA_CA_CERTS={_CONTAINER_CA_BUNDLE}",
                    "-e",
                    f"REQUESTS_CA_BUNDLE={_CONTAINER_CA_BUNDLE}",
                    "-e",
                    f"SSL_CERT_FILE={_CONTAINER_CA_BUNDLE}",
                ]
            )
        elif host_ca:
            logger.warning(
                "%s=%s does not exist; skipping CA bundle mount",
                _CA_BUNDLE_ENV,
                host_ca,
            )

        # ── Extra env vars ──────────────────────────────────────────────
        if env:
            for key, val in env.items():
                if key in _HOST_CONTROL_ENV_KEYS:
                    continue
                cmd.extend(["-e", f"{key}={val}"])

        cmd.extend([image, "/opt/gaia2-init-entrypoint.sh"])

        result = self._run(cmd)
        container_id = result.stdout.strip()
        logger.info(
            "Started container %s (%s) from %s",
            container_name,
            container_id[:12],
            image,
        )
        return container_id

    def exec(
        self,
        container_id: str,
        command: list[str],
        *,
        user: str | None = None,
    ) -> str:
        cmd = [*self._rt, "exec"]
        if user:
            cmd.extend(["-u", user])
        cmd.append(container_id)
        cmd.extend(command)
        return self._run(cmd).stdout

    def copy_from(
        self,
        container_id: str,
        src_path: str,
        dst_path: str,
    ) -> None:
        """Copy a file from the container to the host.

        Uses ``podman cp`` which works on both running and stopped containers.
        """
        self._run([*self._rt, "cp", f"{container_id}:{src_path}", dst_path])

    def stop(self, container_id: str) -> None:
        """Stop and remove the container."""
        try:
            self._run([*self._rt, "stop", "-t", "5", container_id])
        except subprocess.CalledProcessError:
            logger.warning("Container %s already stopped", container_id[:12])
        try:
            self._run([*self._rt, "rm", "-f", container_id])
        except subprocess.CalledProcessError:
            pass
        logger.info("Removed container %s", container_id[:12])
