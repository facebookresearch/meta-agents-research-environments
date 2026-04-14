#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""MITM HTTP CONNECT proxy for faketime-safe TLS.

Accepts CONNECT requests from Node.js (running under libfaketime), terminates
TLS with a per-host certificate signed by a build-time CA, and opens a real
TLS connection to the upstream server using the system clock.

This allows Node.js to run with LD_PRELOAD=libfaketime so that new Date()
returns simulated time, while outbound HTTPS still validates certificates
against real wall-clock time.

Usage:
    python3 tls_proxy.py --port 8099 \
        --ca-cert /opt/tls-proxy/ca.crt --ca-key /opt/tls-proxy/ca.key

The proxy MUST be started without LD_PRELOAD (real system time).
"""

import argparse
import os
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse


def log(msg: str) -> None:
    print(f"[tls-proxy] {msg}", flush=True)


def _resolve_upstream_https_proxy() -> tuple[str, int, str] | None:
    """Return (host, port, label) for the upstream CONNECT proxy, if any."""
    raw = os.environ.get("UPSTREAM_HTTPS_PROXY", "").strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "http").lower()
    if scheme != "http":
        log(
            f"unsupported UPSTREAM_HTTPS_PROXY scheme {scheme!r}; using direct upstream"
        )
        return None

    host = parsed.hostname
    port = parsed.port or 80
    if not host:
        log("invalid UPSTREAM_HTTPS_PROXY; using direct upstream")
        return None

    return host, port, raw


def _open_upstream_socket(hostname: str, port: int) -> tuple[socket.socket, str]:
    """Open a TCP tunnel to the upstream target, optionally via HTTP CONNECT proxy."""
    proxy = _resolve_upstream_https_proxy()
    if not proxy:
        return socket.create_connection((hostname, port), timeout=30), "direct"

    proxy_host, proxy_port, proxy_label = proxy
    upstream_sock = socket.create_connection((proxy_host, proxy_port), timeout=30)
    connect_req = (
        f"CONNECT {hostname}:{port} HTTP/1.1\r\n"
        f"Host: {hostname}:{port}\r\n"
        "User-Agent: gaia2-tls-proxy/1.0\r\n"
        "Proxy-Connection: Keep-Alive\r\n"
        "\r\n"
    ).encode("ascii")
    upstream_sock.sendall(connect_req)

    buf = b""
    while b"\r\n\r\n" not in buf and len(buf) < 65536:
        chunk = upstream_sock.recv(4096)
        if not chunk:
            upstream_sock.close()
            raise ConnectionError("upstream proxy closed during CONNECT")
        buf += chunk

    status_line = buf.split(b"\r\n", 1)[0].decode(errors="replace").strip()
    if not (
        status_line.startswith("HTTP/1.1 200") or status_line.startswith("HTTP/1.0 200")
    ):
        upstream_sock.close()
        raise ConnectionError(f"upstream proxy CONNECT failed: {status_line}")

    return upstream_sock, f"proxy={proxy_label}"


# ---------------------------------------------------------------------------
# Certificate generation (thread-safe)
# ---------------------------------------------------------------------------

_cert_cache: dict[str, ssl.SSLContext] = {}
_cert_lock = threading.Lock()
_host_key_path: str | None = None


def _ensure_host_key(host_key_arg: str | None, tmp_dir: str) -> str:
    global _host_key_path
    if _host_key_path is not None:
        return _host_key_path
    if host_key_arg and os.path.exists(host_key_arg):
        _host_key_path = host_key_arg
        return host_key_arg
    key_path = os.path.join(tmp_dir, "host.key")
    subprocess.run(
        ["/usr/bin/openssl", "genrsa", "-out", key_path, "2048"],
        check=True,
        capture_output=True,
    )
    _host_key_path = key_path
    return key_path


def _get_server_ctx(
    hostname: str,
    ca_cert: str,
    ca_key: str,
    host_key_arg: str | None,
    tmp_dir: str,
) -> tuple[ssl.SSLContext, bool]:
    """Return (server SSLContext, was_cached) for hostname."""
    if hostname in _cert_cache:
        return _cert_cache[hostname], True

    with _cert_lock:
        if hostname in _cert_cache:
            return _cert_cache[hostname], True

        host_key = _ensure_host_key(host_key_arg, tmp_dir)

        csr_path = os.path.join(tmp_dir, f"{hostname}.csr")
        subprocess.run(
            [
                "/usr/bin/openssl",
                "req",
                "-new",
                "-key",
                host_key,
                "-out",
                csr_path,
                "-subj",
                f"/CN={hostname}",
            ],
            check=True,
            capture_output=True,
        )

        ext_path = os.path.join(tmp_dir, f"{hostname}.ext")
        with open(ext_path, "w") as f:
            f.write(
                f"subjectAltName=DNS:{hostname}\n"
                f"basicConstraints=CA:FALSE\n"
                f"keyUsage=digitalSignature,keyEncipherment\n"
                f"extendedKeyUsage=serverAuth\n"
            )

        cert_path = os.path.join(tmp_dir, f"{hostname}.crt")
        serial_path = os.path.join(tmp_dir, "ca.srl")
        sign_env = dict(os.environ)
        sign_env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1"
        sign_env["FAKETIME"] = "2020-01-01 00:00:00"
        sign_env.pop("FAKETIME_TIMESTAMP_FILE", None)
        sign_env.pop("FAKETIME_NO_CACHE", None)
        subprocess.run(
            [
                "/usr/bin/openssl",
                "x509",
                "-req",
                "-in",
                csr_path,
                "-CA",
                ca_cert,
                "-CAkey",
                ca_key,
                "-CAserial",
                serial_path,
                "-CAcreateserial",
                "-out",
                cert_path,
                "-extfile",
                ext_path,
                "-days",
                "7300",
            ],
            check=True,
            capture_output=True,
            env=sign_env,
        )

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert_path, host_key)
        _cert_cache[hostname] = ctx
        return ctx, False


# ---------------------------------------------------------------------------
# Bidirectional relay
# ---------------------------------------------------------------------------


def _relay(src: socket.socket, dst: socket.socket, label: str) -> None:
    """Copy data from src to dst until EOF or error."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Client handler (runs in its own thread)
# ---------------------------------------------------------------------------


def _handle_client(
    client_sock: socket.socket,
    ca_cert: str,
    ca_key: str,
    host_key_arg: str | None,
    tmp_dir: str,
) -> None:
    upstream_sock = None
    upstream_ssl = None
    try:
        client_sock.settimeout(30)
        buf = b""
        # Read CONNECT request line + headers
        while b"\r\n\r\n" not in buf and len(buf) < 8192:
            chunk = client_sock.recv(4096)
            if not chunk:
                client_sock.close()
                return
            buf += chunk

        first_line = buf.split(b"\r\n", 1)[0].decode(errors="replace").strip()

        # Health check
        if first_line.upper().startswith("GET /HEALTHZ"):
            resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nok"
            client_sock.sendall(resp)
            client_sock.close()
            return

        # Parse CONNECT
        parts = first_line.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            client_sock.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            client_sock.close()
            return

        target = parts[1]
        if ":" in target:
            hostname, port_str = target.rsplit(":", 1)
            port = int(port_str)
        else:
            hostname = target
            port = 443

        # Generate per-host cert
        server_ctx, cert_cached = _get_server_ctx(
            hostname,
            ca_cert,
            ca_key,
            host_key_arg,
            tmp_dir,
        )

        # Send 200 Connection Established
        client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        # TLS handshake with the client (server-side)
        client_ssl = server_ctx.wrap_socket(client_sock, server_side=True)

        # Connect to real upstream directly or through the host-side proxy relay.
        t0 = time.monotonic()
        upstream_ctx = ssl.create_default_context()
        upstream_sock, upstream_route = _open_upstream_socket(hostname, port)
        upstream_ssl = upstream_ctx.wrap_socket(upstream_sock, server_hostname=hostname)

        log(
            f"CONNECT {hostname}:{port} ({upstream_route}, cert={'cached' if cert_cached else 'new'}, upstream={time.monotonic() - t0:.1f}s)"
        )

        # Set non-blocking for relay (or use threads)
        client_ssl.settimeout(180)
        upstream_ssl.settimeout(180)

        # Bidirectional relay with two threads
        t1 = threading.Thread(
            target=_relay,
            args=(client_ssl, upstream_ssl, f"{hostname}:c→u"),
            daemon=True,
        )
        t2 = threading.Thread(
            target=_relay,
            args=(upstream_ssl, client_ssl, f"{hostname}:u→c"),
            daemon=True,
        )
        t1.start()
        t2.start()
        t1.join(timeout=300)
        t2.join(timeout=300)

    except Exception as e:
        log(f"handler error: {e}")
    finally:
        for sock in (upstream_ssl, upstream_sock, client_sock):
            try:
                sock.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main server
# ---------------------------------------------------------------------------


def main(port: int, ca_cert: str, ca_key: str, host_key: str | None) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="tls-proxy-")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(128)
    actual_port = server.getsockname()[1]
    log(f"listening on 127.0.0.1:{actual_port}")
    # Write actual port to file so entrypoint can read it
    port_file = os.environ.get("TLS_PROXY_PORT_FILE")
    if port_file:
        with open(port_file, "w") as f:
            f.write(str(actual_port))

    try:
        while True:
            client_sock, addr = server.accept()
            t = threading.Thread(
                target=_handle_client,
                args=(client_sock, ca_cert, ca_key, host_key, tmp_dir),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MITM HTTP CONNECT proxy for faketime-safe TLS"
    )
    parser.add_argument("--port", type=int, default=8099, help="Local listen port")
    parser.add_argument("--ca-cert", required=True, help="Path to CA certificate")
    parser.add_argument("--ca-key", required=True, help="Path to CA private key")
    parser.add_argument(
        "--host-key", default=None, help="Path to pre-generated host key"
    )
    args = parser.parse_args()

    main(args.port, args.ca_cert, args.ca_key, args.host_key)
