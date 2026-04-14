# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""CLI execution bridge for ENV actions.

Converts ``(app_name, fn_name, args)`` tuples to CLI subprocess calls and
runs them.  Used by the adapter's ``/execute_action`` route (Docker mode) and
directly by ``eventd`` in file mode.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

from gaia2_cli.app_registry import (
    APP_TO_CLI as _APP_TO_CLI,
)
from gaia2_cli.app_registry import (
    CLI_TO_MODULE as _CLI_TO_MODULE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Click flag introspection cache
# ---------------------------------------------------------------------------

_flag_params_cache: dict[tuple[str, str], set[str]] = {}


def get_flag_params(cli_name: str, subcmd: str) -> set[str]:
    """Return the set of ``--option`` names that are Click flags for *subcmd*.

    Lazily imports the CLI module and inspects the Click command's params.
    Results are cached so each (cli, subcmd) pair is inspected only once.
    """
    key = (cli_name, subcmd)
    if key not in _flag_params_cache:
        flags: set[str] = set()
        mod_name = _CLI_TO_MODULE.get(cli_name)
        if mod_name:
            try:
                import importlib

                import click

                mod = importlib.import_module(mod_name)
                cli_group = getattr(mod, "cli", None)
                if cli_group:
                    cmd = cli_group.commands.get(subcmd)
                    if cmd:
                        for param in cmd.params:
                            if isinstance(param, click.Option) and param.is_flag:
                                for opt in param.opts:
                                    if opt.startswith("--"):
                                        flags.add(opt)
            except Exception:
                pass
        _flag_params_cache[key] = flags
    return _flag_params_cache[key]


def build_cli_cmd(
    app_name: str, fn_name: str, args: dict[str, Any]
) -> list[str] | None:
    """Convert an ENV action to a CLI command list."""
    cli = _APP_TO_CLI.get(app_name)
    if not cli:
        return None

    # snake_case → kebab-case for Click subcommand
    subcmd = fn_name.replace("_", "-")
    cmd = [cli, subcmd]

    flag_params = get_flag_params(cli, subcmd)

    for key, value in args.items():
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and option in flag_params:
            # Click flag: bare --option when True, omit when False
            if value:
                cmd.append(option)
        elif isinstance(value, (list, dict)):
            cmd.extend([option, json.dumps(value)])
        elif value is not None:
            cmd.extend([option, str(value)])

    return cmd


def run_cli(cmd: list[str], event_id: str = "", state_dir: str = "") -> str | None:
    """Execute a CLI command and return stdout, or *None* on failure.

    If *event_id* is provided it is passed via ``GAIA2_EVENT_ID`` so that
    ``log_action()`` includes it in the events.jsonl entry.
    """
    import shlex

    logger.info("ENV→CLI: %s", shlex.join(cmd))
    env = {**os.environ}
    if state_dir:
        env["GAIA2_STATE_DIR"] = state_dir
    if event_id:
        env["GAIA2_EVENT_ID"] = event_id
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode != 0:
            logger.error(
                "ENV_EXEC_FAILED cmd=%s event_id=%s exit=%d stderr=%s",
                shlex.join(cmd),
                event_id,
                result.returncode,
                result.stderr.strip(),
            )
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.error(
            "ENV_EXEC_FAILED cmd=%s event_id=%s reason=timeout",
            shlex.join(cmd),
            event_id,
        )
        return None
    except Exception as e:
        logger.error(
            "ENV_EXEC_FAILED cmd=%s event_id=%s reason=%s",
            shlex.join(cmd),
            event_id,
            e,
        )
        return None


def execute_cli_action(
    app: str,
    action: str,
    args: dict[str, Any],
    event_id: str = "",
    state_dir: str = "",
) -> dict[str, Any]:
    """Execute an ENV action via CLI subprocess.

    Returns ``{"ok": True, "result": "<stdout>"}`` on success,
    ``{"ok": False, "error": "<reason>"}`` on failure.
    """
    cmd = build_cli_cmd(app, action, args)
    if not cmd:
        return {"ok": False, "error": f"no_cli_mapping for app={app}"}
    result = run_cli(cmd, event_id=event_id, state_dir=state_dir)
    if result is not None:
        return {"ok": True, "result": result}
    return {"ok": False, "error": f"cli_failed for {app}.{action}"}
