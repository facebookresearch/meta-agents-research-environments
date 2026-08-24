# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""GAIA2 JSON parsing helpers."""

from __future__ import annotations

import json


def extract_initial_prompt(data_json: str) -> str | None:
    """Extract the initial user prompt from a scenario's JSON data."""
    data = json.loads(data_json)
    for event in data["events"]:
        if "send_message_to_agent" in event["action"]["action_id"]:
            for arg in event["action"]["args"]:
                if arg["name"] == "content":
                    return arg["value"]
    return None


def extract_expected_response(data_json: str) -> str | None:
    """Extract an expected response from a scenario's JSON data.

    Returns the content of the first ``send_message_to_user`` event found.
    Event order in the data JSON is NOT chronological, so this is mainly
    useful for quick reporting (e.g. counting changed responses).

    For translation, use :func:`extract_all_expected_responses` which returns
    every ``send_message_to_user`` event with its index.
    """
    data = json.loads(data_json)
    for event in data["events"]:
        action = event["action"]
        func = action.get("function", action.get("action_id", ""))
        if func == "send_message_to_user":
            for arg in action.get("args", []):
                if arg["name"] == "content":
                    return arg["value"]
    return None


def extract_all_expected_responses(data_json: str) -> list[tuple[int, str]]:
    """Extract all ``send_message_to_user`` oracle event contents.

    Returns a list of ``(event_idx, content)`` tuples — one per oracle event
    that sends a message to the user.  These are the intermediate and final
    expected responses in a scenario.
    """
    data = json.loads(data_json)
    responses: list[tuple[int, str]] = []
    for event_idx, event in enumerate(data["events"]):
        action = event["action"]
        func = action.get("function", action.get("action_id", ""))
        if func == "send_message_to_user":
            for arg in action.get("args", []):
                if arg["name"] == "content":
                    responses.append((event_idx, arg["value"]))
    return responses


def extract_oracle_events(data_json: str) -> list[dict]:
    """Walk all events in the data JSON and extract oracle event info.

    Returns a list of dicts, one per event:
        {event_idx, app, function, args: {name: value}}
    """
    data = json.loads(data_json)
    oracle_events = []
    for event_idx, event in enumerate(data["events"]):
        action = event["action"]

        args = {}
        for arg in action.get("args") or []:
            args[arg["name"]] = arg["value"]

        oracle_events.append(
            {
                "event_idx": event_idx,
                "app": action.get("app", ""),
                "function": action.get("function", ""),
                "args": args,
            }
        )
    return oracle_events


def replace_event_arg(
    data_json: str, event_idx: int, arg_name: str, new_value: str
) -> str:
    """Replace a single arg value at a specific event index."""
    data = json.loads(data_json)
    event = data["events"][event_idx]
    for arg in event["action"]["args"]:
        if arg["name"] == arg_name:
            arg["value"] = new_value
            break
    return json.dumps(data, ensure_ascii=False)


def extract_completed_events(data_json: str) -> list[dict]:
    """Extract completed event info from the ``completed_events`` key.

    Same shape as :func:`extract_oracle_events` but reads from
    ``data["completed_events"]`` instead of ``data["events"]``.
    Returns an empty list when the key is missing.
    """
    data = json.loads(data_json)
    completed = data.get("completed_events", [])
    results = []
    for event_idx, event in enumerate(completed):
        action = event.get("action")
        if action is None:
            continue

        args = {}
        for arg in action.get("args") or []:
            args[arg["name"]] = arg["value"]

        results.append(
            {
                "event_idx": event_idx,
                "app": action.get("app", ""),
                "function": action.get("function", ""),
                "args": args,
            }
        )
    return results


def replace_completed_event_arg(
    data_json: str, event_idx: int, arg_name: str, new_value: str
) -> str:
    """Replace a single arg value in ``completed_events`` at *event_idx*."""
    data = json.loads(data_json)
    event = data["completed_events"][event_idx]
    for arg in event["action"]["args"]:
        if arg["name"] == arg_name:
            arg["value"] = new_value
            break
    return json.dumps(data, ensure_ascii=False)
