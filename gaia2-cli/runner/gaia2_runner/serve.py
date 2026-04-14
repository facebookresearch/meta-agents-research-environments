# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Live trace viewer server.

Serves the trace viewer HTML and auto-regenerates it when new
scenarios or runs complete.

Single-run mode (directory has scenario subdirs with result.json)::

    python -m gaia2_runner.cli serve --output-dir /tmp/eval-search --port 44100

Multi-run mode (directory has run subdirs with run_config.json)::

    python -m gaia2_runner.cli serve --output-dir /tmp/gaia2-runs --port 44100
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from gaia2_runner.trace_viewer import generate_all, generate_runs_index

_REFRESH_FILE_NAMES = frozenset(
    {
        "run_config.json",
        "results.jsonl",
        "result.json",
        "trace.jsonl",
        "events.jsonl",
        "daemon_status.json",
    }
)


def _looks_like_run_dir(path: Path) -> bool:
    return path.is_dir() and (
        path.name.startswith("run_") or (path / "run_config.json").exists()
    )


def _is_multi_run(output_dir: str) -> bool:
    """Detect whether a directory is a multi-run traces root.

    We only treat a directory as multi-run when its immediate children are
    run directories (``run_1``, ``run_2``, ...) or when its immediate children
    are experiment directories that themselves contain run directories.

    This avoids misclassifying single-run outputs that mirror split folders
    (for example ``search/scenario_x/...``) as multi-run.
    """
    p = Path(output_dir)
    child_dirs = [d for d in p.iterdir() if d.is_dir()]
    if any(_looks_like_run_dir(d) for d in child_dirs):
        return True
    return any(any(_looks_like_run_dir(sub) for sub in d.iterdir()) for d in child_dirs)


def _refresh_fingerprint(output_dir: str | Path) -> str:
    """Return a stable fingerprint of artifacts relevant to live refresh.

    The fingerprint tracks nested trace/state files, not just final
    ``result.json`` files, so the live viewer can refresh while scenarios are
    still in progress.
    """
    root = Path(output_dir)
    digest = hashlib.sha1()

    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if path.is_file() and path.name in _REFRESH_FILE_NAMES:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))

    return digest.hexdigest()


def _regenerate_loop(
    output_dir: str,
    multi_run: bool,
    interval: int = 15,
    *,
    generate_immediately: bool = False,
) -> None:
    """Regenerate the trace viewer every *interval* seconds."""
    last_hash = ""
    first_iteration = generate_immediately
    while True:
        try:
            force = first_iteration
            first_iteration = False
            current = _refresh_fingerprint(output_dir)
            if force or current != last_hash:
                if multi_run:
                    generate_runs_index(output_dir)
                else:
                    generate_all(output_dir)
                last_hash = current
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Regeneration error: %s", exc, exc_info=True
            )
        time.sleep(interval)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live trace viewer server")
    parser.add_argument(
        "output_dir",
        help="Directory with trace artifacts (single run or multi-run root)",
    )
    parser.add_argument("--port", type=int, default=44100)
    parser.add_argument(
        "--interval", type=int, default=15, help="Seconds between regeneration checks"
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    multi_run = _is_multi_run(output_dir)

    if multi_run:
        print(f"Multi-run mode: {output_dir}")
        generate_runs_index(output_dir)
    else:
        print(f"Single-run mode: {output_dir}")
        generate_all(output_dir)

    t = threading.Thread(
        target=_regenerate_loop,
        args=(output_dir, multi_run, args.interval),
        daemon=True,
    )
    t.start()

    os.chdir(output_dir)

    import socket

    class DualStackHTTPServer(HTTPServer):
        address_family = socket.AF_INET6

        def server_bind(self) -> None:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            super().server_bind()

    server = DualStackHTTPServer(("::", args.port), SimpleHTTPRequestHandler)
    hostname = socket.getfqdn() or socket.gethostname()
    print(f"Serving trace viewer at http://{hostname}:{args.port}")
    print(f"Auto-regenerating every {args.interval}s when trace artifacts change")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
