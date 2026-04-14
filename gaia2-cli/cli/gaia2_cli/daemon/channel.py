# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Channel adapter for gaia2-eventd daemon ↔ agent communication.

**FileChannelAdapter**: File-based communication via ``notifications.jsonl``
and ``agent_responses.jsonl``.  The daemon writes notifications to the file;
the agent (or replay harness) reads them.  HTTP delivery to the agent gateway
is handled separately via ``--notify-url``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileChannelAdapter:
    """File-based channel adapter for agent ↔ daemon communication."""

    def __init__(
        self,
        notifications_path: str | Path,
        responses_path: str | Path,
        events_jsonl_path: str | Path,
    ) -> None:
        self.notifications_path = Path(notifications_path)
        self.responses_path = Path(responses_path)
        self.events_jsonl_path = Path(events_jsonl_path)
        self._last_notification_offset: int = 0

    # ------------------------------------------------------------------
    # Agent → Daemon (responses)
    # ------------------------------------------------------------------

    def send_response(self, content: str) -> None:
        """Send an agent response, logging it as send_message_to_user.

        This does two things:
        1. Appends to agent_responses.jsonl (for the daemon to track)
        2. Appends a send_message_to_user event to events.jsonl
           (the canonical turn boundary signal)
        """
        timestamp = time.time()

        # Write to agent_responses.jsonl
        response = {
            "content": content,
            "timestamp": timestamp,
        }
        with open(self.responses_path, "a") as f:
            f.write(json.dumps(response) + "\n")

        # Write send_message_to_user event to events.jsonl
        event = {
            "t": timestamp,
            "app": "AgentUserInterface",
            "fn": "send_message_to_user",
            "args": {"content": content},
            "w": True,
            "ret": None,
        }
        # Include simulated time from faketime.rc if available
        try:
            with open("/tmp/faketime.rc") as f:
                event["sim_t"] = f.read().strip()
        except OSError:
            pass
        with open(self.events_jsonl_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    # ------------------------------------------------------------------
    # Daemon → Agent (notifications)
    # ------------------------------------------------------------------

    def read_notifications(self) -> list[dict[str, Any]]:
        """Read new notifications since the last read.

        Returns a list of notification dicts, each with at least a 'type'
        field ('user_message', 'env_action', etc.).
        """
        if not self.notifications_path.exists():
            return []

        notifications: list[dict[str, Any]] = []
        with open(self.notifications_path, "r") as f:
            f.seek(self._last_notification_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    notifications.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            self._last_notification_offset = f.tell()

        return notifications

    def wait_for_notification(
        self, poll_interval: float = 0.5, timeout: float = 300.0
    ) -> list[dict[str, Any]]:
        """Block until new notifications arrive or timeout.

        Returns the new notifications, or empty list on timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            notifications = self.read_notifications()
            if notifications:
                return notifications
            time.sleep(poll_interval)
        return []
