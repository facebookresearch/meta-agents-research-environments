# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import pickle
import time
from collections import deque
from threading import Thread
from typing import Any, AsyncGenerator, Callable, Type

import strawberry

from are.simulation.agents.agent_log import (
    ActionLog,
    AgentUserInterfaceLog,
    BaseAgentLog,
    EndTaskLog,
    EnvironmentNotificationLog,
    ErrorLog,
    FactsLog,
    FinalAnswerLog,
    LLMInputLog,
    LLMOutputFactsLog,
    LLMOutputPlanLog,
    LLMOutputThoughtActionLog,
    ObservationLog,
    PlanLog,
    RationaleLog,
    RefactsLog,
    ReplanLog,
    StepLog,
    StopLog,
    SubagentLog,
    SystemPromptLog,
    TaskLog,
    ThoughtLog,
    ToolCallLog,
)
from are.simulation.agents.multimodal import Attachment
from are.simulation.gui.server.constants import FILES_PATH
from are.simulation.gui.server.graphql.types import (
    AgentLogForGraphQL,
    AgentLogTypeForGraphQL,
    AttachmentForGraphQL,
)
from are.simulation.types import EnvironmentState, Hint
from are.simulation.utils import make_serializable

graphql_cache: dict[str, dict[str, Any]] = {}
attachment_cache: dict[str, dict[int, tuple[str, int]]] = {}
logger = logging.getLogger(__name__)


def save_attachment_to_disk(
    log_id: str,
    attachment_index: int,
    cache_dir: str,
    hosting_root: str,
    attachment: Attachment,
) -> tuple[str, int]:
    """
    Save an attachment to disk and return the URL and size.
    :param log_id: The ID of the log containing the attachment
    :param attachment_index: The position of the attachment in the log.attachments list
    :param attachment: The attachment to save
    :returns: A tuple containing the URL of the saved attachment and its size in bytes
    """
    if log_id in attachment_cache and attachment_index in attachment_cache[log_id]:
        return attachment_cache[log_id][attachment_index]

    os.makedirs(cache_dir, exist_ok=True)

    extension = ".bin"
    guessed_extension = mimetypes.guess_extension(attachment.mime)
    if guessed_extension:
        extension = guessed_extension

    filename = f"{log_id}_{attachment_index}{extension}"
    filepath = os.path.join(cache_dir, filename)

    decoded_data = base64.b64decode(attachment.base64_data)
    size = len(decoded_data)

    with open(filepath, "wb") as f:
        f.write(decoded_data)

    url = f"{hosting_root}/{filename}"

    if log_id not in attachment_cache:
        attachment_cache[log_id] = {}
    attachment_cache[log_id][attachment_index] = (url, size)

    return (url, size)


def clear_graphql_cache(session_id: str):
    if session_id in graphql_cache:
        del graphql_cache[session_id]


@strawberry.type
class EnvironmentSubscriptionState:
    apps_state_json: str | None = None
    event_log_json: str | None = None
    initial_event_queue_json: str | None = None
    env_state: EnvironmentState | None = None
    hints: list[Hint] | None = None
    world_logs: list[AgentLogForGraphQL] | None = None
    environment_time: float | None = None


@strawberry.type
class Subscription:
    server = None  # ARESimulationGuiServer | None - Can't type this because of circular dependency

    @strawberry.subscription
    async def environment_subscription_state(
        self, session_id: str
    ) -> AsyncGenerator[EnvironmentSubscriptionState, None]:
        """
        Asynchronous generator function that yields the state of the environment
        for a given session. It monitors changes in the environment state and
        updates the client with the latest state.

        :param session_id: The unique identifier for the session.
        :type session_id: str
        :return: An asynchronous generator yielding EnvironmentSubscriptionState objects.
        :rtype: AsyncGenerator[EnvironmentSubscriptionState, None]
        """

        def _run_apps_state_json(session_id: str):
            """
            Internal function that runs in a separate thread to monitor and
            update the state of applications, event logs, and environment state
            for a given session.

            :param session_id: The unique identifier for the session.
            :type session_id: str
            """
            try:
                if Subscription.server is None:
                    raise ValueError("Subscription.server is not initialized.")
                are_simulation_instance = (
                    Subscription.server.get_or_create_are_simulation(session_id)
                )

                cache_dir = os.path.join(are_simulation_instance.tmpdir, "cache")
                hosting_root = f"{FILES_PATH}/{session_id}/cache"

                while keep_alive:
                    try:
                        # Create a new EnvironmentSubscriptionState object to store new changes
                        changed_state = EnvironmentSubscriptionState()

                        # Handle any changes in the event log
                        event_log = make_serializable(
                            {
                                "event_log": are_simulation_instance.env.event_log.to_dict()
                            }
                        )

                        if update_graphql_cache(session_id, event_log, "event_log"):
                            changed_state.event_log_json = json.dumps(event_log)

                        # Handle any changes in the app state
                        apps_modified = any(
                            app.is_state_modified
                            for app in are_simulation_instance.env.apps.values()  # type: ignore
                        )

                        if changed_state.event_log_json is not None or apps_modified:
                            apps_state = are_simulation_instance.env.get_apps_state()
                            changed_state.apps_state_json = json.dumps(
                                make_serializable(apps_state)
                            )

                        # Handle any changes in the initial event queue
                        initial_event_queue = make_serializable(
                            [
                                event.to_dict()
                                for event in are_simulation_instance.scenario.events  # type: ignore
                            ]
                        )
                        if update_graphql_cache(
                            session_id, initial_event_queue, "initial_event_queue"
                        ):
                            changed_state.initial_event_queue_json = json.dumps(
                                initial_event_queue
                            )

                        # Handle any changes in the environment state
                        env_state = are_simulation_instance.env.state
                        if update_graphql_cache(session_id, env_state, "env_state"):
                            changed_state.env_state = env_state

                        # Handle any changes in agent logs
                        # We need to use an array as default because having no logs is a valid state (e.g. on soft reset)
                        world_logs: list[BaseAgentLog] | None = (
                            are_simulation_instance.get_world_logs()
                        ) or []
                        if update_graphql_cache(
                            session_id,
                            world_logs,
                            "world_logs",
                            check_length_only=True,
                        ):
                            world_logs_for_graphql = get_world_logs_for_graphql(
                                world_logs, cache_dir, hosting_root
                            )
                            changed_state.world_logs = world_logs_for_graphql

                        # Handle any changes in hints
                        hints = are_simulation_instance.scenario.hints or []
                        if update_graphql_cache(session_id, hints, "hints"):
                            changed_state.hints = hints

                        # Reset the is_state_modified flag for all apps
                        for app in are_simulation_instance.env.apps.values():  # type: ignore
                            app.is_state_modified = False

                        # Handle any changes in environment time
                        environment_time = (
                            are_simulation_instance.env.time_manager.time()
                        )
                        cached_environment_time = graphql_cache.get(session_id, {}).get(
                            "environment_time"
                        )
                        if (
                            cached_environment_time is None
                            or environment_time - cached_environment_time > 1
                        ):
                            graphql_cache.setdefault(session_id, {})[
                                "environment_time"
                            ] = environment_time
                            changed_state.environment_time = environment_time

                        # If any of the states have changed, add the changed_state to the update queue
                        if any(
                            [
                                changed_state.apps_state_json,
                                changed_state.event_log_json,
                                changed_state.initial_event_queue_json,
                                changed_state.env_state,
                                changed_state.hints,
                                changed_state.world_logs,
                                changed_state.environment_time,
                            ]
                        ):
                            update_queue.append(changed_state)
                        else:
                            # If no states have changed, wait for 0.25 seconds before checking again
                            time.sleep(0.25)
                    except Exception as e:
                        logger.exception(f"An error occurred: {e}")
                        continue  # Restart the loop on error so client can get the latest state
            except Exception as e:
                logger.exception(f"An error occurred: {e}")

        t: Thread | None = None
        try:
            if Subscription.server is None:
                raise ValueError("Subscription.server is not initialized.")
            keep_alive = True
            update_queue = deque()
            # State change detection is done in a separate thread to avoid blocking.
            t = Thread(
                target=_run_apps_state_json,
                args=(session_id,),
                name=f"Subscription_{session_id}",
            )
            t.start()

            # Wait for the session to become available.
            while not Subscription.server.session_manager.session_exists(session_id):
                await asyncio.sleep(0.1)
            while Subscription.server.session_manager.session_exists(session_id):
                while not update_queue:
                    await asyncio.sleep(0.1)
                yield update_queue.popleft()
        except asyncio.CancelledError:
            logger.debug(f"Subscription for session {session_id} was cancelled.")
        except Exception as e:
            logger.exception(f"An error occurred: {e}")
        finally:
            keep_alive = False
            if t is not None and t.is_alive():
                t.join()

            clear_graphql_cache(session_id)
            logger.debug(f"Cleaned up Subscription for session {session_id}.")
            # Always yield to allow asyncio proper cleanup.
            yield EnvironmentSubscriptionState()


def update_graphql_cache(
    session_id: str,
    obj: Any,
    key_prefix: str,
    /,
    check_length_only: bool | None = False,
) -> bool:
    """
    Checks if the given object has changed since the last time it was checked and updates the graphql_cache.

    :param session_id: The session identifier for the current environment.
    :type session_id: str
    :param obj: The object whose state is being tracked.
    :type obj: Any
    :param key_prefix: A prefix used to identify the type of state being updated.
    :type key_prefix: str
    :param check_length_only: If True, only the length
    :type check_length_only: bool | None
    :return: True if the object has changed and the cache was updated, False otherwise.
    :rtype: bool
    """
    if isinstance(obj, (str, int, float, bool)):
        # Directly handle string values without JSON conversion
        if graphql_cache.get(session_id, {}).get(key_prefix) != obj:
            graphql_cache.setdefault(session_id, {})[key_prefix] = obj
            return True

    elif isinstance(obj, list) and check_length_only:
        # Check and cache only the length of the list
        obj_length = len(obj)
        cached_length = graphql_cache.get(session_id, {}).get(f"{key_prefix}_length")
        if cached_length != obj_length:
            graphql_cache.setdefault(session_id, {})[f"{key_prefix}_length"] = (
                obj_length
            )
            return True

    else:
        # Compute the hash of the object
        obj_hash = hashlib.sha256(pickle.dumps(obj)).hexdigest()
        # Check if the hash has changed
        if graphql_cache.get(session_id, {}).get(f"{key_prefix}_hash") != obj_hash:
            # Update the cache
            graphql_cache.setdefault(session_id, {})[f"{key_prefix}_hash"] = obj_hash
            return True
    return False


def process_attachments(
    log_id: str, cache_dir: str, hosting_root: str, attachments: list[Attachment] | None
) -> list[AttachmentForGraphQL] | None:
    """
    Process attachments for a log and return them in GraphQL format.

    :param log_id: The ID of the log containing attachments
    :param attachments: List of attachments to process or None
    :return: A list of AttachmentForGraphQL objects or None if no attachments
    """
    if not attachments:
        return None

    result = []
    for i, attachment in enumerate(attachments):
        url, size = save_attachment_to_disk(
            log_id, i, cache_dir, hosting_root, attachment
        )
        result.append(
            AttachmentForGraphQL(
                length=size,
                mime=attachment.mime,
                url=url,
            )
        )
    return result


def _create_agent_log_for_graphql(
    log: BaseAgentLog,
    graphql_type: AgentLogTypeForGraphQL,
    content: str | None = None,
    group_id: str | None = None,
    attachments: list[AttachmentForGraphQL] | None = None,
    exception: str | None = None,
) -> AgentLogForGraphQL:
    """Helper function to create AgentLogForGraphQL objects with common fields."""
    return AgentLogForGraphQL(
        id=log.id,
        timestamp=log.timestamp,
        type=graphql_type,
        content=content,
        group_id=group_id,
        attachments=attachments,
        exception=exception,
    )


def get_world_logs_for_graphql(
    world_logs: list[BaseAgentLog],
    cache_dir: str,
    hosting_root: str,
) -> list[AgentLogForGraphQL]:
    world_logs_for_graphql = []
    current_group_id = None
    step_counter = 0

    world_logs.sort(key=lambda log: log.timestamp)

    # Log type mappings for logs that require simple content extraction
    # These logs have straightforward content that can be directly mapped to GraphQL
    # without special processing (no JSON serialization, attachments, or exceptions)
    simple_log_mappings: dict[
        Type[BaseAgentLog],
        tuple[AgentLogTypeForGraphQL, Callable[[BaseAgentLog], str | None]],
    ] = {
        # System and LLM communication logs
        SystemPromptLog: (
            AgentLogTypeForGraphQL.SYSTEM_PROMPT,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        LLMOutputThoughtActionLog: (
            AgentLogTypeForGraphQL.LLM_OUTPUT_THOUGHT_ACTION,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        LLMOutputPlanLog: (
            AgentLogTypeForGraphQL.LLM_OUTPUT_PLAN,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        LLMOutputFactsLog: (
            AgentLogTypeForGraphQL.LLM_OUTPUT_FACTS,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        # Agent reasoning and planning logs
        RationaleLog: (
            AgentLogTypeForGraphQL.RATIONALE,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        PlanLog: (AgentLogTypeForGraphQL.PLAN, lambda log: log.content),  # type: ignore[attr-defined]
        FactsLog: (
            AgentLogTypeForGraphQL.FACTS,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        ReplanLog: (
            AgentLogTypeForGraphQL.REPLAN,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        RefactsLog: (
            AgentLogTypeForGraphQL.REFACTS,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        ThoughtLog: (
            AgentLogTypeForGraphQL.THOUGHT,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
        # Tool and action logs
        ToolCallLog: (
            AgentLogTypeForGraphQL.TOOL_CALL,
            lambda log: (
                log.get_content_for_llm()
            ),  # Uses method to format tool call data
        ),
        # Final output logs
        FinalAnswerLog: (
            AgentLogTypeForGraphQL.FINAL_ANSWER,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
    }

    # Log types that require JSON serialization of their content
    # These logs contain complex data structures (dicts, lists) that need to be
    # serialized to JSON strings for GraphQL transmission
    json_serialized_mappings: dict[
        Type[BaseAgentLog], tuple[AgentLogTypeForGraphQL, Callable[[BaseAgentLog], Any]]
    ] = {
        # LLM input contains structured message data that needs JSON serialization
        LLMInputLog: (
            AgentLogTypeForGraphQL.LLM_INPUT,
            lambda log: log.content,  # type: ignore[attr-defined]
        ),
    }

    # Log types that contain file attachments (images, documents, etc.)
    # These require special processing to save attachments to disk and generate URLs
    attachment_log_types: set[Type[BaseAgentLog]] = {
        TaskLog,  # User tasks may include attached files or images
        ObservationLog,  # Tool observations may include screenshots or file outputs
    }

    # Log types that should be completely ignored in the UI
    # These are internal logs that don't provide value to end users
    skip_log_types: set[Type[BaseAgentLog]] = {
        StopLog,  # Internal signal for stopping execution - not user-relevant
        ActionLog,  # Low-level action details - too verbose for UI display
    }

    # Log types that are not yet supported in the UI but may be in the future
    # These generate warnings to help track what functionality is missing
    warn_log_types: set[Type[BaseAgentLog]] = {
        AgentUserInterfaceLog,  # UI interaction logs - future feature
        EnvironmentNotificationLog,  # Environment event notifications - future feature
    }

    for log in world_logs:
        # Handle StepLog separately (special case)
        if isinstance(log, StepLog):
            step_counter += 1
            current_group_id = log.id
            world_logs_for_graphql.append(
                _create_agent_log_for_graphql(
                    log,
                    AgentLogTypeForGraphQL.STEP,
                    content=f"STEP {log.iteration}",
                    group_id=None,
                )
            )
            continue

        # Handle SubagentLog separately (completely different logic)
        if isinstance(log, SubagentLog):
            log_id = log.id
            world_logs_for_graphql.append(
                _create_agent_log_for_graphql(
                    log,
                    AgentLogTypeForGraphQL.SUBAGENT,
                    content=f"SUBAGENT ({log.name})",
                    group_id=current_group_id,
                )
            )

            subworld_logs = get_world_logs_for_graphql(
                log.children, cache_dir, hosting_root
            )
            for subagent_log in subworld_logs:
                subagent_log.is_subagent = True
                if subagent_log.group_id is None:
                    subagent_log.group_id = log_id
                world_logs_for_graphql.append(subagent_log)
            continue

        # Handle ErrorLog separately (has exception field)
        if isinstance(log, ErrorLog):
            world_logs_for_graphql.append(
                _create_agent_log_for_graphql(
                    log,
                    AgentLogTypeForGraphQL.ERROR,
                    content=log.error,
                    group_id=current_group_id,
                    exception=log.exception,
                )
            )
            continue

        # Handle EndTaskLog separately (no content)
        if isinstance(log, EndTaskLog):
            world_logs_for_graphql.append(
                _create_agent_log_for_graphql(
                    log,
                    AgentLogTypeForGraphQL.END_TASK,
                    group_id=current_group_id,
                )
            )
            continue

        # Handle TaskLog and ObservationLog (with attachments)
        if type(log) in attachment_log_types:
            attachments = process_attachments(
                log.id, cache_dir, hosting_root, getattr(log, "attachments", None)
            )
            graphql_type = (
                AgentLogTypeForGraphQL.TASK
                if isinstance(log, TaskLog)
                else AgentLogTypeForGraphQL.OBSERVATION
            )
            group_id = None if isinstance(log, TaskLog) else current_group_id

            world_logs_for_graphql.append(
                _create_agent_log_for_graphql(
                    log,
                    graphql_type,
                    content=getattr(log, "content", ""),
                    group_id=group_id,
                    attachments=attachments,
                )
            )
            continue

        # Handle logs that need JSON serialization
        log_type = type(log)
        if log_type in json_serialized_mappings:
            graphql_type, content_getter = json_serialized_mappings[log_type]
            content = json.dumps(make_serializable(content_getter(log)))
            world_logs_for_graphql.append(
                _create_agent_log_for_graphql(
                    log,
                    graphql_type,
                    content=content,
                    group_id=current_group_id,
                )
            )
            continue

        # Handle simple logs with direct content mapping
        if log_type in simple_log_mappings:
            graphql_type, content_getter = simple_log_mappings[log_type]
            content = content_getter(log)
            world_logs_for_graphql.append(
                _create_agent_log_for_graphql(
                    log,
                    graphql_type,
                    content=content,
                    group_id=current_group_id,
                )
            )
            continue

        # Handle logs to skip
        if log_type in skip_log_types:
            continue

        # Handle logs to warn about
        if log_type in warn_log_types:
            log_name = log_type.__name__
            logger.warning(f"{log_name} is not supported yet in the UI")
            continue

        # Unknown log type
        logger.error(f"Unknown log type: {log_type} cannot be returned in GraphQL")
        continue

    # Patch logs to display correctly a subagent for Facts and Plan
    i = 0
    while i < len(world_logs_for_graphql) - 1:
        current_log = world_logs_for_graphql[i]
        next_log = world_logs_for_graphql[i + 1]

        if current_log.type == AgentLogTypeForGraphQL.LLM_INPUT and next_log.type in (
            AgentLogTypeForGraphQL.LLM_OUTPUT_FACTS,
            AgentLogTypeForGraphQL.LLM_OUTPUT_PLAN,
        ):
            subagent_id = f"subagent_{current_log.id}"
            subagent_timestamp = current_log.timestamp
            subagent_group_id = current_log.group_id

            subagent_name = (
                "facts"
                if next_log.type == AgentLogTypeForGraphQL.LLM_OUTPUT_FACTS
                else (
                    "plan"
                    if next_log.type == AgentLogTypeForGraphQL.LLM_OUTPUT_PLAN
                    else "unknown"
                )
            )

            subagent_log = AgentLogForGraphQL(
                id=subagent_id,
                timestamp=subagent_timestamp,
                type=AgentLogTypeForGraphQL.SUBAGENT,
                content=f"SUBAGENT ({subagent_name})",
                group_id=subagent_group_id,
            )

            world_logs_for_graphql.insert(i, subagent_log)

            world_logs_for_graphql[i + 1].group_id = subagent_id
            world_logs_for_graphql[i + 2].group_id = subagent_id

            i += 3
        else:
            i += 1

    return world_logs_for_graphql
