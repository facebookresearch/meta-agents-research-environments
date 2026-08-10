# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Container launcher for Gaia2 eval containers.

Provides ContainerLauncher ABC, LocalLauncher (podman/docker), and
ApptainerLauncher (for HPC nodes where an overlayfs /scratch blocks rootless
podman).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

logger: logging.Logger = logging.getLogger(__name__)

_RED = "\033[31m"
_RESET = "\033[0m"

# Default port for the local proxy relay (host process → upstream proxy).
_RELAY_PORT = 18888
_PROXY_RELAY_ENV = "GAIA2_PROXY_RELAY_URL"
_CA_BUNDLE_ENV = "GAIA2_CA_BUNDLE"
# Root of the per-instance scratch tree used by ApptainerLauncher.
_APPTAINER_SCRATCH_ENV = "GAIA2_APPTAINER_SCRATCH"
_CONTAINER_CA_BUNDLE = "/etc/ssl/certs/gaia2-host-ca-bundle.crt"
_HOST_CONTROL_ENV_KEYS = frozenset({_PROXY_RELAY_ENV, _CA_BUNDLE_ENV})

# Proxy env vars are computed by the launch() proxy/CA block; they must not be
# copied verbatim from the inherited environment (which would clobber them).
_PROXY_ENV_KEYS = frozenset(
    {"http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"}
)

# Provider-SDK base URL / auth env vars that MUST be stripped from the container
# env before agent startup. Some host tooling points these at a session-scoped
# loopback proxy that only exists on the machine that set them; if they leak
# into the container the agent's SDK redirects every LLM call to a port where
# nothing is listening and the run aborts after a couple of events.
_SDK_REDIRECT_ENV_KEYS = frozenset(
    {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "OPENAI_BASE_URL"}
)

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


def _pipe_socket(src: socket.socket, dst: socket.socket) -> None:
    """Pump bytes one-way from src to dst until src closes.

    On EOF (or error) it half-closes dst's write side so the peer sees the
    stream end. Used by the proxy relay.
    """
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except Exception:
            pass


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

    def _handle(client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection((proxy_host, proxy_port))
            t1 = threading.Thread(
                target=_pipe_socket, args=(client, upstream), daemon=True
            )
            t2 = threading.Thread(
                target=_pipe_socket, args=(upstream, client), daemon=True
            )
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
            logger.warning(
                "Could not bind proxy relay on 127.0.0.1:%d (port in use); "
                "assuming an existing relay is already listening there",
                listen_port,
            )
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

    def _is_podman_runtime(self) -> bool:
        """Return whether this launcher invokes a Podman CLI."""
        return any(
            os.path.basename(part) in {"podman", "podman-remote"} for part in self._rt
        )

    def _should_publish_adapter_port(
        self,
        *,
        network: str,
        adapter_port: int | None,
    ) -> bool:
        """Use port publishing for local Podman on macOS.

        Podman's ``--network=host`` is the Podman VM's host namespace on
        macOS, not the macOS host. Publishing the adapter port is what makes
        ``127.0.0.1:<port>`` reachable from the runner process.
        """
        return (
            sys.platform == "darwin"
            and self._is_podman_runtime()
            and network == "host"
            and adapter_port is not None
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
        publish_adapter_port = self._should_publish_adapter_port(
            network=network,
            adapter_port=adapter_port,
        )

        cmd = [*self._rt, "run", "-d", f"--name={container_name}"]
        if publish_adapter_port:
            cmd.extend(["-p", f"127.0.0.1:{adapter_port}:{adapter_port}"])
        else:
            cmd.append(f"--network={network}")
        cmd.extend(
            [
                # Workaround for crun seccomp cache permission errors on
                # some kernels (linkat EPERM in seccomp bpf setup).
                "--security-opt",
                "seccomp=unconfined",
                "-v",
                f"{scenario_json_path}:/var/gaia2/custom_scenario.json:ro",
                "--entrypoint",
                "bash",
            ]
        )
        if self._is_podman_runtime():
            # Pass host supplementary groups into the user namespace so
            # bind-mounted files on group-readable shared filesystems
            # (e.g. Lustre with mode 0770) are accessible without world perms.
            cmd.extend(["--group-add", "keep-groups"])
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
                if key in _SDK_REDIRECT_ENV_KEYS:
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


# ── Apptainer implementation ────────────────────────────────────────────


class ApptainerLauncher(ContainerLauncher):
    """Container launcher using apptainer (SquashFS .sif on FUSE).

    Designed for cluster nodes where rootless podman fails because /scratch is
    overlayfs and the vfs storage driver can't chown into it.
    Apptainer mounts a read-only .sif via FUSE, runs containers as
    --fakeroot inside a user namespace (mapping host UID → root inside),
    and keeps all per-container writes in tmpfs — no driver, no graphroot,
    no chown.

    The runner's ContainerRunner doesn't need to know which launcher it
    holds: launch()/exec()/copy_from()/stop() preserve the same semantics
    as LocalLauncher.

    Per-instance scratch (daemon state, the writable ``/var/gaia2/state``
    view, ``/tmp``) lives under ``$GAIA2_APPTAINER_SCRATCH/$USER/apptainer``,
    defaulting to ``$TMPDIR`` and finally ``/scratch``. Point
    ``GAIA2_APPTAINER_SCRATCH`` at node-local fast storage on sites that
    don't mount ``/scratch``.
    """

    def __init__(self, image_sif: str | None = None) -> None:
        sif = image_sif or os.environ.get("GAIA2_OC_SIF", "").strip()
        if not sif:
            raise ValueError(
                "ApptainerLauncher needs a .sif path: pass image_sif=... "
                "or set GAIA2_OC_SIF in the environment"
            )
        if not os.path.isfile(sif):
            raise FileNotFoundError(f"Apptainer .sif not found: {sif}")
        self.image_sif = self._maybe_stage_to_local_scratch(sif)

    @staticmethod
    def _scratch_root() -> Path:
        """Root for per-instance scratch and staged .sif copies.

        ``/scratch`` is only the last-resort default — many sites don't mount
        it, so honour ``GAIA2_APPTAINER_SCRATCH`` and then ``TMPDIR``.
        """
        base = os.environ.get(_APPTAINER_SCRATCH_ENV, "").strip() or os.environ.get(
            "TMPDIR", "/scratch"
        )
        user = os.environ.get("USER", "user")
        return Path(base) / user / "apptainer"

    @staticmethod
    def _maybe_stage_to_local_scratch(sif: str) -> str:
        # A .sif on shared network storage is a bottleneck: every parallel
        # apptainer instance fans out FUSE reads of the same file over the
        # network, which starves the in-container daemons at startup. Staging
        # one node-local copy lets all instances share a warm page cache.
        #
        # Off by default so the launcher stays safe where the scratch root
        # doesn't exist or the .sif is already local. Opt in with
        # GAIA2_OC_SIF_STAGE_LOCAL=1.
        if os.environ.get("GAIA2_OC_SIF_STAGE_LOCAL", "").strip() != "1":
            return sif
        local_dir = ApptainerLauncher._scratch_root()
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Cannot create %s for .sif staging (%s); using original path %s",
                local_dir,
                e,
                sif,
            )
            return sif
        local_sif = local_dir / Path(sif).name
        src_mtime = os.path.getmtime(sif)
        if local_sif.is_file() and os.path.getmtime(local_sif) >= src_mtime:
            logger.debug("Apptainer .sif already staged at %s", local_sif)
            return str(local_sif)
        # The flock is required, not defensive: without it concurrent runners on
        # the same node race on the same .tmp filename, then os.replace swaps a
        # half-written SquashFS into place and every later `apptainer instance
        # start` fails. Losers of the lock re-check the mtime and short-circuit.
        lock_path = local_dir / f"{Path(sif).name}.stage.lock"
        with open(lock_path, "w") as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            if local_sif.is_file() and os.path.getmtime(local_sif) >= src_mtime:
                logger.debug("Apptainer .sif already staged at %s", local_sif)
                return str(local_sif)
            tmp_path = local_sif.with_suffix(local_sif.suffix + ".tmp")
            logger.info("Staging .sif from %s to %s", sif, local_sif)
            shutil.copyfile(sif, tmp_path)
            os.replace(tmp_path, local_sif)
        return str(local_sif)

    def _run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        logger.debug("Running: %s", " ".join(args))
        return subprocess.run(
            args, capture_output=True, text=True, check=True, **kwargs
        )

    @staticmethod
    def _instance_scratch_dir(container_id: str) -> Path:
        return ApptainerLauncher._scratch_root() / container_id

    @staticmethod
    def _instance_env_file(scratch: Path) -> Path:
        """Host path of the bind-mounted env file holding the container secrets.

        Kept in its own ``secrets/`` subdirectory (mode 0o700) rather than at
        the scratch root: the scratch root has to stay traversable/1777 for the
        ``/tmp`` and ``/var/gaia2/state`` bind sources used by the unprivileged
        in-container users, whereas this file is only ever read by the fakeroot
        (uid 0 == launching host UID) wrapper shell.
        """
        return scratch / "secrets" / "env.sh"

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
        # `image` arg (e.g. "localhost/gaia2-oc:latest") is ignored — apptainer
        # needs a .sif path, supplied at construction time.
        container_name = self._container_name(scenario_json_path)

        # Per-instance scratch for daemon state + the writable view on
        # /var/gaia2/state; --writable-tmpfs (per-instance since apptainer
        # 1.5.1) covers the rest of the rootfs. Do NOT swap it for a
        # per-instance --overlay <dir>: where the host rootfs is itself an
        # overlay, the nested overlay strips exec perms on /usr/bin/bash for
        # non-root uids and the entrypoint's `su -s /usr/bin/bash ...` gets
        # EACCES.
        scratch = self._instance_scratch_dir(container_name)
        (scratch / "state").mkdir(parents=True, exist_ok=True)
        # Per-instance /tmp, mode 1777 so the in-container agent user can
        # write under it.
        tmp_dir = scratch / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(tmp_dir, 0o1777)

        env_file_host = self._instance_env_file(scratch)

        cmd = [
            "apptainer",
            "instance",
            "start",
            "--fakeroot",
            "--writable-tmpfs",
            "--bind",
            f"{tmp_dir}:/tmp",
            "--bind",
            f"{scratch}/state:/var/gaia2/state",
            "--bind",
            f"{scenario_json_path}:/var/gaia2/custom_scenario.json:ro",
            "--bind",
            f"{env_file_host}:/var/gaia2/env.sh:ro",
        ]

        for volume in extra_volumes or ():
            cmd.extend(["--bind", volume])

        # Env vars passed via `--env` on `instance start` do NOT propagate to
        # processes started by `apptainer exec instance://...` — that's a
        # known apptainer quirk (env scoped to the exec session). So we
        # collect them here and apply them on the entrypoint exec below.
        env_pairs: list[tuple[str, str]] = []

        if adapter_port is not None:
            env_pairs.append(("GAIA2_ADAPTER_PORT", str(adapter_port)))
        if gateway_port is not None:
            env_pairs.append(("OPENCLAW_GATEWAY_PORT", str(gateway_port)))
            env_pairs.append(("OPENCLAW_GATEWAY_URL", f"ws://127.0.0.1:{gateway_port}"))

        for key, val in self._build_provider_env(
            image, provider=provider, model=model, api_key=api_key, env=env
        ):
            env_pairs.append((key, val))

        # Proxy / CA: same logic as LocalLauncher, minus the in-process relay
        # (the relay listens on the host's 127.0.0.1 and apptainer's host
        # network passthrough — the default — makes it reachable directly).
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
            for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                env_pairs.append((k, relay_url))
            env_pairs.append(("NO_PROXY", no_proxy_value))
            env_pairs.append(("no_proxy", no_proxy_value))
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
                    env_pairs.append((proxy_var, val))

        host_ca = _resolve_ca_bundle_path(env)
        if host_ca and os.path.isfile(host_ca):
            cmd.extend(["--bind", f"{host_ca}:{_CONTAINER_CA_BUNDLE}:ro"])
            for k in ("NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
                env_pairs.append((k, _CONTAINER_CA_BUNDLE))
        elif host_ca:
            logger.warning(
                "%s=%s does not exist; skipping CA bundle mount",
                _CA_BUNDLE_ENV,
                host_ca,
            )

        if env:
            for key, val in env.items():
                if key in _HOST_CONTROL_ENV_KEYS:
                    continue
                # Proxy vars are owned exclusively by the proxy/CA block above;
                # skip them so inherited raw values can't clobber the computed
                # ones.
                if key in _PROXY_ENV_KEYS:
                    continue
                # See _SDK_REDIRECT_ENV_KEYS.
                if key in _SDK_REDIRECT_ENV_KEYS:
                    continue
                env_pairs.append((key, val))

        cmd.extend([self.image_sif, container_name])

        # Write the env file BEFORE instance start — it is bind-mounted at start
        # time. APPTAINERENV_* / os.environ on the later `apptainer exec` does
        # not reach the child, so the wrapper sources this file instead.
        #
        # SECURITY: env_pairs carries live API keys, so this is a clear-text
        # secret store on a possibly shared filesystem. _write_private_file
        # creates it 0o600 inside a 0o700 directory with no window at looser
        # perms, and stop() deletes it even when scratch is kept.
        # 0o600 still lets the container read it: --fakeroot maps the launching
        # host UID (the file's owner) to uid 0 in the namespace, and the only
        # reader is the root wrapper shell below. The unprivileged in-container
        # users never read it — the entrypoint re-exports what they need into
        # its own 0o600 file.
        _write_private_file(
            env_file_host,
            "".join(f"export {key}={_shell_quote(val)}\n" for key, val in env_pairs),
        )

        self._run(cmd)

        log_path = scratch / "entrypoint.out"
        # Apptainer-only fixups, all required before the entrypoint runs:
        # 1. --writable-tmpfs copy-up under --fakeroot leaves /var/gaia2 and
        #    /tmp root-owned and restrictive, so the gaia2 daemons can neither
        #    traverse state nor write logs.
        # 2. The image's setuid-gaia2 `gaia2-exec` wrapper (how the agent user
        #    reaches gaia2-owned state) is a no-op here: the kernel ignores
        #    setuid bits on unprivileged FUSE mounts, and apptainer mounts the
        #    rootfs via fuse-overlayfs. Every state read would return EACCES.
        #    Replace the privilege split with shared-group access: add agent to
        #    the gaia2 group and relax state perms (g+rwX files, setgid dirs).
        #    This trades away the "agent can't read state out-of-band"
        #    guarantee, which is acceptable for single-run benchmark eval.
        # 3. `umask 002` is what actually grants g+w to files created at
        #    runtime; without it .lock files come out 0644 and the other user's
        #    flock(LOCK_EX) gets EACCES. Patch both `su` calls — `su` resets
        #    the umask.
        # The sed edits the in-image entrypoint in place; the overlay is
        # per-instance so nothing leaks across containers.
        wrapper = (
            "chown gaia2:gaia2 /var/gaia2 && "
            "chmod 1777 /tmp && "
            "usermod -aG gaia2 agent && "
            "sed -i "
            "-e 's|chmod 700 /var/gaia2|chmod 750 /var/gaia2|' "
            '-e \'s|chmod -R go-rwx "$STATE_DIR"|'
            'chmod -R g+rwX,o-rwx "$STATE_DIR" \\&\\& '
            'find "$STATE_DIR" -type d -exec chmod g+s {} +|\' '
            "-e 's|su -s /usr/bin/bash gaia2 -c \"|"
            "su -s /usr/bin/bash gaia2 -c \"umask 002; |' "
            "-e 's|su -s /usr/bin/bash agent -c \"|"
            "su -s /usr/bin/bash agent -c \"umask 002; |' "
            "/opt/gaia2-init-entrypoint.sh && "
            "umask 002 && "
            # The Dockerfile's `ARG/ENV http_proxy` freezes the build host's
            # proxy into the .sif, and it propagates all the way into the
            # agent's LLM client, which then dials a proxy that usually isn't
            # routable from the run host. Unset it BEFORE sourcing env.sh so
            # relay mode (which writes real proxy vars there) is unaffected.
            "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && "
            "set -a && . /var/gaia2/env.sh && set +a && "
            "exec /opt/gaia2-init-entrypoint.sh"
        )
        with open(log_path, "ab") as logf:
            subprocess.Popen(
                [
                    "apptainer",
                    "exec",
                    f"instance://{container_name}",
                    "bash",
                    "-c",
                    wrapper,
                ],
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

        logger.info(
            "Started apptainer instance %s from %s",
            container_name,
            self.image_sif,
        )
        return container_name

    def exec(
        self,
        container_id: str,
        command: list[str],
        *,
        user: str | None = None,
    ) -> str:
        # Apptainer has no `-u <user>`. Inside the container we run as
        # fakeroot (uid 0), so when the caller asks for `gaia2` we wrap in
        # `su -s /bin/bash gaia2 -c "..."`. The default (user=None) runs
        # as fakeroot, which is what the runner wants for `cat`, `ls`, etc.
        cmd = ["apptainer", "exec", f"instance://{container_id}"]
        if user:
            quoted = " ".join(_shell_quote(arg) for arg in command)
            cmd.extend(["su", "-s", "/bin/bash", user, "-c", quoted])
        else:
            cmd.extend(command)
        return self._run(cmd).stdout

    def copy_from(
        self,
        container_id: str,
        src_path: str,
        dst_path: str,
    ) -> None:
        """Copy a file from the container to the host.

        Two layers:
        1. /var/gaia2/state is bind-mounted from host scratch — read it
           directly with shutil.copy (no container needed). Works even
           after the instance has stopped.
        2. Everything else (notably /tmp which lives in writable-tmpfs):
           use `apptainer exec cat` for files, or tar-stream for dirs.
        """
        scratch = self._instance_scratch_dir(container_id)
        # /var/gaia2/state/* is bind-mounted from host scratch, but the daemon
        # writes those files 0o600 owned by an apptainer subuid, so from the
        # host the shutil shortcut raises PermissionError before it can even
        # stat them. Inside the --fakeroot instance `cat` runs as uid 0 and
        # reads them fine, so force state paths down the exec branch.
        force_exec = src_path.startswith("/var/gaia2/state")
        host_mirror = (
            None if force_exec else self._map_container_path_to_host(src_path, scratch)
        )
        logger.debug(
            "copy_from src=%s host_mirror=%s exists=%s scratch=%s force_exec=%s",
            src_path,
            host_mirror,
            host_mirror.exists() if host_mirror else None,
            scratch,
            force_exec,
        )
        if host_mirror is not None and host_mirror.exists():
            if host_mirror.is_dir():
                shutil.copytree(host_mirror, dst_path, dirs_exist_ok=True)
            else:
                Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(host_mirror, dst_path)
            return

        # Fallback: pull via exec. Use tar to handle both files and dirs.
        is_dir_check = subprocess.run(
            [
                "apptainer",
                "exec",
                f"instance://{container_id}",
                "test",
                "-d",
                src_path,
            ],
            capture_output=True,
        )
        is_dir = is_dir_check.returncode == 0

        if is_dir:
            os.makedirs(dst_path, exist_ok=True)
            tar_proc = subprocess.Popen(
                [
                    "apptainer",
                    "exec",
                    f"instance://{container_id}",
                    "tar",
                    "-C",
                    os.path.dirname(src_path.rstrip("/")) or "/",
                    "-cf",
                    "-",
                    os.path.basename(src_path.rstrip("/")),
                ],
                stdout=subprocess.PIPE,
            )
            untar = subprocess.run(
                ["tar", "-C", dst_path, "--strip-components=1", "-xf", "-"],
                stdin=tar_proc.stdout,
                check=True,
            )
            tar_proc.wait()
            if tar_proc.returncode != 0 or untar.returncode != 0:
                raise RuntimeError(f"copy_from(dir) failed for {src_path}")
        else:
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            out = subprocess.run(
                [
                    "apptainer",
                    "exec",
                    f"instance://{container_id}",
                    "cat",
                    src_path,
                ],
                capture_output=True,
                check=True,
            )
            Path(dst_path).write_bytes(out.stdout)

    @staticmethod
    def _map_container_path_to_host(container_path: str, scratch: Path) -> Path | None:
        """Map a bind-mounted container path back to its host scratch path."""
        if container_path.startswith("/var/gaia2/state"):
            rel = container_path[len("/var/gaia2/state") :].lstrip("/")
            return scratch / "state" / rel if rel else scratch / "state"
        return None

    def stop(self, container_id: str) -> None:
        """Stop the apptainer instance and remove its scratch."""
        try:
            self._run(["apptainer", "instance", "stop", "--timeout", "5", container_id])
        except subprocess.CalledProcessError:
            logger.warning("Apptainer instance %s already stopped", container_id)
        scratch = self._instance_scratch_dir(container_id)
        # Always shred the credential file, even when the rest of scratch is
        # kept for debugging — it holds clear-text API keys.
        env_file_host = self._instance_env_file(scratch)
        try:
            env_file_host.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "Could not remove credential file %s: %s", env_file_host, exc
            )
        if os.environ.get("GAIA2_OC_KEEP_SCRATCH") == "1":
            logger.info(
                "Stopped apptainer instance %s (scratch kept at %s)",
                container_id,
                scratch,
            )
            return
        try:
            shutil.rmtree(scratch, ignore_errors=True)
        except Exception as exc:
            logger.debug("Failed to clean scratch for %s: %s", container_id, exc)
        logger.info("Stopped apptainer instance %s", container_id)


def _write_private_file(path: Path, content: str) -> None:
    """Write `content` to `path` as an owner-only (0o600) file.

    Used for files holding credentials. Two properties matter on a shared
    cluster filesystem, where every other user on the node (and on the shared
    mount) can otherwise read them:

    * the parent directory is created 0o700, and re-tightened if it already
      existed with looser perms;
    * the file never exists at looser-than-0o600 perms. ``os.open`` with
      ``O_CREAT|O_EXCL`` and mode 0o600 creates it restricted from the very
      first instant (the mode is umask-masked, which can only remove bits, and
      ``O_EXCL`` guarantees we are not inheriting a pre-existing file's mode).
      A stale file from a previous run of the same container id is removed
      first rather than truncated in place.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(content)


def _shell_quote(s: str) -> str:
    """POSIX shell quoting for embedding values in env.sh / args in `su -c`.

    Delegates to ``shlex.quote`` so *every* shell-significant character is
    covered. A hand-rolled version that quotes only the common metacharacters
    is not enough: some providers issue API keys containing a pipe, and an
    unquoted ``export GAIA2_JUDGE_API_KEY=<key>`` line is then parsed as a
    *pipeline* when the entrypoint sources env.sh (``set -a && . env.sh``),
    leaving the key empty and making litellm raise "Missing credentials" on
    every call.
    """
    return shlex.quote(s)
