# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Judge-only task text for user turns the agent receives in another form.

When a scenario delivers a user turn to the agent as something other than its
text (e.g. native audio, with a neutral carrier string in the scenario's
``content`` arg), the judge must still grade against what the user actually
asked.  That original text lives in a side file the agent can never read::

    {"version": 1, "turns": {"<USER event id>": {"text": "<original prompt>"}}}

Inside the container it sits at ``$GAIA2_JUDGE_CONTEXT`` (default
``/var/gaia2/judge_context.json``), owned by ``gaia2`` with mode ``0600``.  On
the host it is a sidecar ``<scenario stem>.judge_context.json`` next to the
scenario file: the launcher mounts it read-only at
``STAGED_JUDGE_CONTEXT_PATH`` and the init entrypoint installs the private
copy.  Host-side ``ScenarioLoader`` (offline re-judging) reads the sidecar
directly.

Error messages name paths, event ids and types, never the text: they end up in
``gaia2-eventd.log``, which the agent user can read.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

JUDGE_CONTEXT_ENV = "GAIA2_JUDGE_CONTEXT"
DEFAULT_JUDGE_CONTEXT_PATH = "/var/gaia2/judge_context.json"
# Where a launcher mounts the host sidecar read-only. The init entrypoint
# installs it at the path gaia2-eventd reads, owned by gaia2 with mode 0600;
# /var/gaia2 itself is closed to the agent.
STAGED_JUDGE_CONTEXT_PATH = "/var/gaia2/judge_context.staged.json"
JUDGE_CONTEXT_VERSION = 1
SIDECAR_SUFFIX = ".judge_context.json"

_LABEL_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class JudgeContextError(ValueError):
    """The judge context is unreadable, malformed, or does not fit the scenario."""


def safe_label(key: str) -> str:
    """Return *key* for an error message, or a placeholder if it is not id-like.

    Keys of a judge context are event ids; anything else could be prompt text
    put in the wrong place, so it is described rather than echoed.
    """
    if _LABEL_RE.match(key):
        return key
    return f"<{len(key)}-char key>"


def _type_name(value: Any) -> str:
    """JSON-ish name of *value*'s type, for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def judge_context_sidecar_path(scenario_path: str | os.PathLike[str]) -> Path:
    """Return the host path of the judge context for *scenario_path*.

    ``/data/search/s1.json`` → ``/data/search/s1.judge_context.json``.
    """
    path = Path(scenario_path)
    return path.with_name(path.stem + SIDECAR_SUFFIX)


def resolve_judge_context_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the in-container path: ``$GAIA2_JUDGE_CONTEXT`` or the default."""
    env = os.environ if environ is None else environ
    return Path(env.get(JUDGE_CONTEXT_ENV) or DEFAULT_JUDGE_CONTEXT_PATH)


def parse_judge_context(data: Any) -> dict[str, str]:
    """Validate a decoded judge context and return ``{event_id: text}``."""
    if not isinstance(data, dict):
        raise JudgeContextError(f"must be a JSON object, got {_type_name(data)}")
    expected = {"version", "turns"}
    if set(data) != expected:
        missing = sorted(expected - set(data))
        unexpected = sorted(safe_label(k) for k in set(data) - expected)
        raise JudgeContextError(
            "must have exactly the keys 'version' and 'turns' "
            f"(missing: {missing or 'none'}, unexpected: {unexpected or 'none'})"
        )

    version = data["version"]
    if type(version) is not int or version != JUDGE_CONTEXT_VERSION:
        shown = version if type(version) is int else _type_name(version)
        raise JudgeContextError(
            f"unsupported version {shown}; this build reads version "
            f"{JUDGE_CONTEXT_VERSION}"
        )

    turns = data["turns"]
    if not isinstance(turns, dict):
        raise JudgeContextError(
            "'turns' must be an object mapping USER event ids to "
            f'{{"text": ...}}, got {_type_name(turns)}'
        )
    if not turns:
        raise JudgeContextError(
            "'turns' is empty; omit the file when no turn needs judge-only text"
        )

    texts: dict[str, str] = {}
    for event_id, entry in turns.items():
        label = safe_label(event_id)
        if not isinstance(entry, dict) or set(entry) != {"text"}:
            raise JudgeContextError(
                f"turn {label}: entry must be an object with exactly one key, 'text'"
            )
        text = entry["text"]
        if not isinstance(text, str):
            raise JudgeContextError(
                f"turn {label}: 'text' must be a string, got {_type_name(text)}"
            )
        if not text.strip():
            raise JudgeContextError(f"turn {label}: 'text' is empty")
        texts[event_id] = text
    return texts


def check_user_turn_ids(texts: Mapping[str, str], user_turn_ids: set[str]) -> None:
    """Check every id in *texts* names one of the scenario's user turns.

    *user_turn_ids* are the event ids of the scenario's USER
    ``send_message_to_agent`` events.  A stale or mistyped id would otherwise
    leave that turn judged against whatever the agent was sent, silently.

    Raises:
        JudgeContextError: Some id is not in *user_turn_ids*.
    """
    unknown = sorted(safe_label(eid) for eid in texts if eid not in user_turn_ids)
    if unknown:
        raise JudgeContextError(
            "judge context names events that are not USER "
            f"send_message_to_agent events of this scenario: {unknown}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json`` object hook: a repeated key would silently keep only the last."""
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise JudgeContextError(f"duplicate key {safe_label(key)}")
        obj[key] = value
    return obj


def load_judge_context(
    path: str | os.PathLike[str],
    *,
    missing_ok: bool = False,
    require_private: bool = False,
) -> dict[str, str] | None:
    """Read and validate the judge context at *path*; return ``{event_id: text}``.

    Returns ``None`` if the file does not exist and *missing_ok* is set.  With
    *require_private*, a file that group or other may access, or that the
    calling user does not own, is refused: the agent shares a group with the
    daemon under some launchers, so anything looser than ``0600`` could hand
    it the original prompt.

    Raises:
        JudgeContextError: The file is missing (without *missing_ok*),
            unreadable, too permissive, or malformed.
    """
    path = Path(path)
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        if missing_ok:
            return None
        raise JudgeContextError(f"judge context not found: {path}") from None
    except OSError as e:
        raise JudgeContextError(
            f"cannot read judge context {path}: {e.strerror or type(e).__name__}"
        ) from None

    try:
        with f:
            if require_private:
                st = os.fstat(f.fileno())
                mode = stat.S_IMODE(st.st_mode)
                if mode & 0o077:
                    raise JudgeContextError(
                        f"mode is {mode:04o}; it must be 0600 and owned by the "
                        "daemon user so the agent cannot read it"
                    )
                if st.st_uid != os.geteuid():
                    raise JudgeContextError(
                        f"owned by uid {st.st_uid}; it must be owned by the "
                        f"daemon user (uid {os.geteuid()}) with mode 0600"
                    )
            try:
                data = json.load(f, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as e:
                raise JudgeContextError(
                    f"not valid JSON ({e.msg} at line {e.lineno} column {e.colno})"
                ) from None
            except UnicodeDecodeError:
                raise JudgeContextError("not valid UTF-8") from None
        return parse_judge_context(data)
    except JudgeContextError as e:
        raise JudgeContextError(f"judge context {path}: {e}") from None


__all__ = [
    "DEFAULT_JUDGE_CONTEXT_PATH",
    "JUDGE_CONTEXT_ENV",
    "JUDGE_CONTEXT_VERSION",
    "JudgeContextError",
    "SIDECAR_SUFFIX",
    "STAGED_JUDGE_CONTEXT_PATH",
    "check_user_turn_ids",
    "judge_context_sidecar_path",
    "load_judge_context",
    "parse_judge_context",
    "resolve_judge_context_path",
    "safe_label",
]
