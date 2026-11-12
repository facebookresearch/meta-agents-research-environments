# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import copy
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Type, TypeVar

from termcolor import colored

from are.simulation.agents.agent_log import ActionLog
from are.simulation.agents.are_simulation_agent import BaseAgentLog
from are.simulation.apps import INTERNAL_APPS
from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.app import App, Protocol
from are.simulation.apps.reminder import ReminderApp
from are.simulation.apps.system import SystemApp
from are.simulation.notification_system import (
    BaseNotificationSystem,
    Message,
    MessageType,
    NotificationSystem,
    get_content_for_message,
)
from are.simulation.scenarios.scenario import Scenario
from are.simulation.time_manager import TimeManager
from are.simulation.tool_utils import AppTool
from are.simulation.types import (
    AbstractEnvironment,
    AbstractEvent,
    Action,
    AgentActionValidator,
    AgentValidationEvent,
    CompletedEvent,
    ConditionCheckEvent,
    EnvironmentState,
    EnvironmentType,
    Event,
    EventLog,
    EventQueue,
    EventType,
    OracleEvent,
    StopEvent,
    ValidationEvent,
    ValidationException,
)
from are.simulation.utils import get_state_dict, make_serializable, save_jsonl

EVENT_LOOP_DELAY_S = 5

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """
    Configuration for the Environment class.

    This class encapsulates all the parameters required to configure the behavior of the Environment,
    including time management, event loop behavior, and other settings.

    Attributes:
    - `start_time` (float or None): The start time of the environment in seconds. Defaults to 0 if None.
    - `duration` (float or None): The duration of the environment in seconds. If None, the environment runs indefinitely.
    - `time_increment_in_seconds` (int): The time increment between each tick in seconds. Must be >= 1.
    - `oracle_mode` (bool): Indicates if the environment is in oracle mode. In oracle mode, OracleEvents are processed.
    - `dump_dir` (str): Directory to dump environment states. Used only in oracle mode.
    - `exit_when_no_events` (bool): Determines whether to exit the event loop when there are no more events to process.
    If True, the event loop exits once all events are processed, even if the duration hasn't been reached.
    This should be set to True only when running without an agent, as it prevents the event loop from blocking indefinitely in `env.join()`.
    When running with an agent, it should typically be False to allow the agent to generate more events over time.
    - `queue_based_loop` (bool): Controls whether the event loop is based on the queue or time:
    Queue-based event loop: The event queue is polled for events to process, and time jumps to the next event time.
    Time-based event loop: Time is incremented in fixed increments, and events are processed when they are ready.
    - `wait_for_user_input_timeout` (float): Timeout for waiting for user input in seconds.
    - `verbose` (bool): Indicates whether to print event loop ticks to the terminal.
    """

    start_time: float | None = None
    duration: float | None = 60
    time_increment_in_seconds: int = 1
    oracle_mode: bool = False
    dump_dir: str | None = None
    exit_when_no_events: bool = False
    queue_based_loop: bool = False
    wait_for_user_input_timeout: float | None = None
    verbose: bool = True

    def __post_init__(self):
        assert self.time_increment_in_seconds >= 1, (
            "Time increment in seconds must be >= 1"
        )
        # queue based event loop can only be used when in oracle mode. It does not make sense to use it with an agent.
        if self.queue_based_loop and not self.oracle_mode:
            raise ValueError("Queue based event loop can only be used in oracle mode.")
        if self.dump_dir and not self.oracle_mode:
            raise ValueError("States can only be dumped in oracle mode.")


# Generic type for `Environment.get_app_with_class(...)` to constrain the return
# value to the type passed into the function.
AppType = TypeVar("AppType", bound=App)


class Environment(AbstractEnvironment):
    """
    The Environment class is responsible for managing the simulation environment.

    It handles the event loop, time management, app registration, and event processing.
    The environment can run with or without an agent, and its behavior can be configured
    through the EnvironmentConfig class.

    The event loop can operate in two modes:
    1. Time-based: Time is incremented in fixed intervals, and events are processed when ready
    2. Queue-based: The event queue is polled for events, and time jumps to the next event time

    The event loop will exit when one of the following conditions is met:
    - The duration is reached (if specified)
    - The stop event is set (via the stop() method)
    - There are no more events to process (if exit_when_no_events is True)

    This last condition should only be enabled when running without an agent, as it prevents
    the event loop from blocking indefinitely when there are no more events to process.
    When running with an agent, exit_when_no_events should typically be False to allow
    the agent to generate more events over time.
    """

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        environment_type: EnvironmentType = EnvironmentType.UNKNOWN,
        notification_system: BaseNotificationSystem | None = None,
        add_event_to_agent_log: Callable[[CompletedEvent], None] | None = None,
    ):
        if config is None:
            config = EnvironmentConfig()

        # Here events to schedule is a list of the events that will be scheduled when the environment starts
        # There are not put the in the queue yet, but a subset of them will be at start time
        self.events_to_schedule = []
        self.event_log = EventLog()
        self.event_queue = EventQueue()
        self.initial_config = config

        # Time management
        self.time_manager = TimeManager()
        self.start_time = config.start_time if config.start_time is not None else 0
        self.time_manager.reset(start_time=self.start_time)
        self.current_time = self.time_manager.time()
        # Tick corresponds to every time increment made
        self.tick_count = 0
        # The time increment in seconds is the time between each tick
        self.time_increment_in_seconds = config.time_increment_in_seconds
        # The duration in seconds that the environment will run for
        self.duration = config.duration
        self.state = EnvironmentState.SETUP
        self.apps = {}
        self.protocol_to_app: dict[Protocol, App] = {}
        self.exit_when_no_events = config.exit_when_no_events
        # Used to send a stop flag to the event loop to stop
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread = None
        self.oracle_mode = config.oracle_mode
        self.queue_based_loop = config.queue_based_loop
        self.wait_for_user_input_timeout = config.wait_for_user_input_timeout
        self.environment_type = environment_type
        self.verbose = config.verbose
        self.waiting_for_notification = False

        # Register the notification system as an app to give access to notifications by default
        if notification_system is None:
            self.log_debug("Creating default notification system")
            notification_system = NotificationSystem()

        self.notification_system = notification_system
        self.notification_system.initialize(self.time_manager)

        self.agent_action_validators: list[AgentActionValidator] = []
        self.log_info(f"Environment created with config: {config}")

        # Temporary solution to allow for GraphQL caching
        self.graphql_cache: dict[str, Any] = {}
        self.dump_dir = config.dump_dir

        self.add_event_to_agent_log: Callable[[CompletedEvent], None] | None = (
            add_event_to_agent_log
        )
        self.world_logs = []
        self.is_replaying = False

    def set_is_replaying(self, is_replaying: bool):
        self.is_replaying = is_replaying

    def start(self, debug: bool = False):
        """
        Starts internal clock from environment
        """

        if debug:
            logger.setLevel(logging.DEBUG)

        if self.state == EnvironmentState.RUNNING:
            self.log_debug("Environment already running.")
            return

        self.state = EnvironmentState.RUNNING
        self.prepare_events_for_start()

        self.thread = threading.Thread(target=self._event_loop, name="EventLoop")
        self.thread.daemon = True
        self.thread.start()

    def stop(self, final_state: EnvironmentState = EnvironmentState.STOPPED):
        """
        Stop the event loop
        """
        self.stop_event.set()
        self.state = final_state

        if self.notification_system is not None:
            self.notification_system.message_queue.put(
                Message(
                    message_type=MessageType.ENVIRONMENT_STOP,
                    message=f"Environment stopped with state {self.state}",
                    timestamp=datetime.fromtimestamp(
                        self.time_manager.time(), tz=timezone.utc
                    ),
                )
            )

    def join(self):
        """
        Wait for the event loop to finish.

        This method blocks until the event loop thread completes. The event loop
        will complete when one of the following conditions is met:
        1. The duration is reached (if specified)
        2. The stop event is set (via the stop() method)
        3. There are no more events to process (if exit_when_no_events is True)

        When running without an agent, it's important to set exit_when_no_events=True
        in the EnvironmentConfig to ensure this method doesn't block indefinitely.
        When running with an agent, exit_when_no_events should typically be False to
        allow the agent to generate more events over time.
        """
        if self.thread is None:
            raise ValueError("Event loop not started")
        self.thread.join()

    def pause(self):
        """
        Pause the event loop
        """
        # We set the pause flag, and wait for the thread to finish as the event loop will stop
        if self.state != EnvironmentState.RUNNING:
            self.log_debug("Attempt to pause when environment is not running.")
            return
        self.pause_event.set()
        self.time_manager.pause()
        self.state = EnvironmentState.PAUSED

    def resume(self):
        """
        Resume the event loop
        """
        if self.state != EnvironmentState.PAUSED:
            self.log_debug("Attempt to resume when environment is not paused.")
            return
        self.time_manager.resume()
        self.log_debug("Environment is resumed.")
        self.pause_event.clear()
        self.state = EnvironmentState.RUNNING

    def resume_with_offset(self, offset: float):
        """
        Resume the event loop with an offset
        """
        if self.state != EnvironmentState.PAUSED:
            self.log_debug("Attempt to resume when environment is not paused.")
            return
        self.time_manager.add_offset(offset)
        self.time_manager.resume()
        self.log_debug("Environment is resumed.")
        self.pause_event.clear()
        self.state = EnvironmentState.RUNNING
        self.tick()  # We tick once to make sure the time is updated

    def prepare_events_for_start(self):
        """
        Prepare the events for start by scheduling them
        """
        self.log_debug("Preparing events for start")
        self.log_debug(f"Found {len(self.events_to_schedule)} events to schedule")
        for event in self.events_to_schedule:
            if len(event.dependencies) == 0:
                event.compute_absolute_time(self.start_time)
                self.log_debug(
                    f"Event {event.event_id} scheduled at {datetime.fromtimestamp(event.event_time, tz=timezone.utc)}"
                )
                self.event_queue.put(event)

    def delete_all_completed_events(self):
        self.event_log = EventLog()

    def get_apps_state(self) -> Any:
        """
        Get the state of the apps
        """
        apps_to_skip = set(internal_app.name for internal_app in INTERNAL_APPS)
        return {
            "apps": [
                {"app_name": name, **(app.get_state() or {})}
                for name, app in self.apps.items()
                if name not in apps_to_skip
            ]
        }

    def apps_state_json(self) -> str:
        return json.dumps(make_serializable(self.get_apps_state()))

    def event_log_json(self) -> str:
        state_dict = {"event_log": self.event_log.to_dict()}
        return json.dumps(make_serializable(state_dict))

    def get_state(self) -> dict[str, Any]:
        """
        Get a simple dict representation of the environment state.
        With the states of the Apps, event log, event queue, and event triggers.
        """
        state_dict = get_state_dict(
            self,
            [
                "start_time",
                "time_increment_in_seconds",
                "duration",
            ],
        )
        state_dict["event_log"] = self.event_log.to_dict()
        state_dict["event_queue"] = self.event_queue.to_dict()
        # We round this to be able to compare states correctly
        state_dict["current_time"] = int(self.current_time)
        # Add app's name to the item.
        state_dict["apps"] = [
            {"app_name": name, **(app.get_state() or {})}
            for name, app in self.apps.items()
        ]
        return make_serializable(state_dict)

    def reset_app_states(self):
        for app in self.apps.values():
            app.reset()

    def put_last_event_from_log_to_queue(self):
        """
        Put the last event in the log back in the event queue with the same time
        it was executed. This is useful for replay.
        """
        self._reset_event_queue()
        if not self.event_log.past_events.empty():
            last_event: CompletedEvent = self.event_log.past_events[-1]
            del self.event_log.past_events[-1]
            self.event_queue.put(last_event.to_future_event())

    def _event_loop(self):
        """
        Periodically performs simulation steps.
        """
        if self.queue_based_loop:
            self._queue_based_loop()
        else:
            self._time_based_loop()

        self.log_info("Event loop finished")
        # final validation checks to fail the environment if needed
        try:
            self.final_validation_checks()
            if self.state != EnvironmentState.FAILED:
                # If the environment did not fail in the middle of the loop, we can stop it
                self.state = EnvironmentState.STOPPED
        except ValidationException as e:
            self.log_error(f"Final Validation failed with exception: {e}")
            self.state = EnvironmentState.FAILED
        # Send notification that the environment has stopped
        if self.notification_system is not None:
            self.notification_system.message_queue.put(
                Message(
                    message_type=MessageType.ENVIRONMENT_STOP,
                    message=f"Environment stopped with state {self.state}",
                    timestamp=datetime.fromtimestamp(
                        self.time_manager.time(), tz=timezone.utc
                    ),
                )
            )
        self.pause_event.clear()
        self.stop_event.clear()
        self.log_info(
            f"COMPLETED - DURATION {self.time_manager.time_passed()} / TICKS {self.tick_count} / EVENTS {len(self.event_log)}"
        )
        if self.dump_dir is not None:
            self._dump_state("final_state.jsonl")

    def _dump_state(self, filename: str):
        """
        Dump the environment state to a JSONL file.

        Args:
            filename: The name of the file to dump the state to
        """
        assert self.dump_dir is not None, "dump_dir should not be None"
        assert os.path.exists(self.dump_dir), f"{self.dump_dir} does not exist"
        dump_path = os.path.join(self.dump_dir, filename)
        self.log_info(f"Dumping environment state to: {dump_path}")
        save_jsonl([self.get_state()], dump_path, overwrite=True)

    def _queue_based_loop(self):
        """
        Periodically performs simulation steps by polling the event queue for events to process
        """
        while self.get_event_queue_length() > 0 and not self.stop_event.is_set():
            self.log_info(f"Event queue length: {self.get_event_queue_length()}")
            while self.pause_event.is_set():
                self.pause_event.wait()  # Wait here when the environment is paused
            self.jump_to_next_event_time()
            if (
                self.duration is not None
                and self.time_manager.time_passed() > self.duration
            ):
                # If after the jump we have passed the duration, we stop the event loop
                break
            self.tick()

    def _reset_event_queue(self):
        """
        Reset the event queue to prepare for event replay.
        """
        self.event_queue.already_scheduled = set()

    def _time_based_loop(self):
        """
        Periodically performs simulation steps by moving time forward in time increments.

        The loop continues until one of the following conditions is met:
        1. The duration is reached (if specified)
        2. The stop event is set (via the stop() method)
        3. There are no more events to process (if exit_when_no_events is True)

        This last condition should only be enabled when running without an agent,
        as it prevents the event loop from blocking indefinitely when waiting for
        env.join() to return. When running with an agent, exit_when_no_events should
        typically be False to allow the agent to generate more events over time.
        """
        while (
            self.duration is None or self.time_manager.time_passed() <= self.duration
        ) and not self.stop_event.is_set():
            if self.exit_when_no_events and self.get_event_queue_length() == 0:
                self.log_info("No more events to process, exiting")
                break
            while self.pause_event.is_set():
                # Wait here when the environment is paused
                time.sleep(0.1)
            self.tick()
            time.sleep(1)
            # This is to simulate that one second in real time is equivalent to time_increment_in_seconds of simulation time
            self.time_manager.add_offset(self.time_increment_in_seconds - 1)
            self.tick_count += 1
        if self.duration is None:
            self.log_info("Event loop finished because duration is None")
        elif self.time_manager.time_passed() > self.duration:
            self.log_info(
                f"Event loop finished because duration {self.duration} reached (time passed: {self.time_manager.time_passed()})"
            )

    def tick(self):
        """
        Execute one tick of the event loop
            - Get triggered events
            - Get events to process and process them
        """
        # TODO: Change with something wrt validation events

        self.current_time = self.time_manager.time()
        self.log_debug(f"Starting Time Tick {self.tick_count}")

        # We make sure we update the validators tick count before the event loop
        self.increment_validators_tick_count()

        # Handle time_based_notifications
        if self.notification_system is not None:
            self.notification_system.handle_time_based_notifications()

        while True:
            events_to_process = self.event_queue.pop_events_to_process(
                self.current_time
            )
            self.log_debug(f"Found {len(events_to_process)} events to process")

            if len(events_to_process) == 0 or events_to_process is None:
                self.log_debug(
                    f"No events to process any more at tick {self.tick_count}"
                )
                break

            for event in events_to_process:
                try:
                    event_successors = self.process_event(event)
                except ValidationException as e:
                    self.log_error(f"Validation failed with exception: {e}")
                    self.log_error("Stopping environment")
                    self.stop(final_state=EnvironmentState.FAILED)
                    return
                self.log_debug(
                    f"Event {event.event_id} processed - had {len(event_successors)} successors"
                )
                for suc in event_successors:
                    if suc.event_time is None:
                        suc.event_time = event.event_time + suc.event_relative_time
                    self.event_queue.put(suc)  # type: ignore

        # Handle timeout notifications after all events in this tick have been processed
        if self.notification_system is not None:
            self.notification_system.handle_timeout_after_events()

    def wait_for_next_notification(self) -> None:
        """
        Wait for the next notification by processing events until a notification is triggered.
        :param timeout: The timeout timestamp in seconds.
        """

        self.log_info("Wait for notification started")
        self.time_manager.pause()

        system_app = self.get_app_with_class(SystemApp)
        assert system_app is not None, "System app not found"
        timeout_timestamp = system_app.wait_for_notification_timeout.timeout_timestamp  # type: ignore

        try:
            self.waiting_for_notification = True

            while not self.stop_event.is_set():
                next_event_time = self.get_next_event_time()
                next_notification_time = (
                    self.notification_system.get_next_notification_time()
                )
                # filter next event and next notif time to only consider those that are before the timeout
                if next_event_time is not None and next_event_time > timeout_timestamp:
                    next_event_time = None
                if (
                    next_notification_time is not None
                    and next_notification_time > timeout_timestamp
                ):
                    next_notification_time = None

                # Next thing is the timeout --> we exit
                if next_event_time is None and next_notification_time is None:
                    jump_time = timeout_timestamp - self.time_manager.time()
                    self.log_debug(f"Jumping to timestamp {timeout_timestamp}")
                    self.time_manager.add_offset(jump_time)
                    return

                # Next thing is a notification --> we exit
                elif next_event_time is None and next_notification_time is not None:
                    self.time_manager.add_offset(
                        next_notification_time - self.time_manager.time()
                    )
                    self.log_debug(f"Jumping to timestamp {next_notification_time}")
                    return

                # Next thing is an event --> we jump to it
                elif next_event_time is not None and next_notification_time is None:
                    self.time_manager.add_offset(
                        next_event_time - self.time_manager.time()
                    )
                    self.log_debug(f"Jumping to timestamp {next_event_time}")

                # Next thing is an event or a notification
                elif next_event_time is not None and next_notification_time is not None:
                    if next_event_time < next_notification_time:
                        self.time_manager.add_offset(
                            next_event_time - self.time_manager.time()
                        )
                        self.log_debug(f"Jumping to timestamp {next_event_time}")
                    else:
                        self.time_manager.add_offset(
                            next_notification_time - self.time_manager.time()
                        )
                        self.log_debug(f"Jumping to timestamp {next_notification_time}")
                        return

                self.tick()

        finally:
            self.tick()
            self.waiting_for_notification = False
            system_app.reset_wait_for_notification_timeout()
            self.time_manager.resume()
            self.log_info("Wait for notification finished")

    def jump_to_next_event_time(self):
        """
        Skip to the next tick where events should be processed
        Thus skipping all ticks where no events are to be processed
        """
        next_event_time = self.get_next_event_time()
        self.log_info(f"Next event time: {next_event_time}")
        assert next_event_time is not None, "No next event time found"
        assert self.start_time is not None, "Start time is None"

        next_tick = math.ceil(
            (next_event_time - self.start_time) / self.time_increment_in_seconds
        )
        self.log_debug(f"Jumping to next tick {next_tick}")
        self.tick_count = next_tick
        time_jump = next_event_time - self.current_time
        self.log_debug(f"Jumping time by {time_jump} seconds")
        self.time_manager.add_offset(time_jump)

    def get_next_event_time(self) -> float | None:
        """
        Get the next event time from the event queue.
        """
        next_event = self.event_queue.peek()
        if next_event is None:
            return None
        return next_event.event_time

    def process_event(self, event: AbstractEvent) -> list[AbstractEvent]:  # type: ignore
        successors = []

        if type(event) is StopEvent:
            self.log_info("Stop event triggered")
            self.stop()
            return []

        if type(event) is OracleEvent:
            if self.oracle_mode:
                # If we are in Oracle mode, we create an Agent event from the Oracle event
                # This event is handled like any regular Event later in the function
                self.log_debug(f"Oracle event {event.event_id} triggered")
                event: Event = event.make(self).with_type(EventType.AGENT)  # type: ignore
                self.log_info(f"Oracle event: function {str(event.action)}")
            else:
                self.log_debug(
                    f"Oracle event {event.event_id} ignored as oracle_mode = False"
                )
                return []

        if isinstance(event, Event):
            event = self.resolve_arg_placeholders(event, self.event_log.list_view())
            completed_event = event.execute()

            self.log_debug(f"Completed event {completed_event.event_id}")
            self.add_to_log(completed_event)
            successors = self.get_successors(event)  # type: ignore

        elif type(event) is ValidationEvent:
            validation_result, completed_event = event.validate(self)
            self.add_to_log(completed_event)
            if validation_result.success:
                self.log_info(
                    f"Validation {event.event_id} succeeded - {validation_result.success}"
                )
            else:
                self.log_debug(
                    f"Checking validation {event.event_id} - success = {validation_result.success}"
                )
            self.log_debug(
                f"{len(validation_result.achieved_milestones)} milestones achieved so far"
            )
            self.log_debug(
                f"Milestones achieved {validation_result.achieved_milestones}"
            )

            next_event = event.get_next_event(self.time_increment_in_seconds)
            if next_event is not None:
                successors = [next_event]

        elif type(event) is ConditionCheckEvent:
            success, completed_event = event.check(self)
            self.add_to_log(completed_event)
            if success:
                self.log_info(f"Condition {event.event_id} succeeded")
            else:
                self.log_debug(
                    f"Checking condition {event.event_id} - success = {success}"
                )

            if success:
                successors = self.get_successors(event)  # type: ignore

            elif not event.is_timeout():
                # If the check failed, we reschedule another one for after the schedule_every_ticks
                successors = [
                    event.get_next_check_event(self.time_increment_in_seconds)
                ]

            elif event.is_timeout():
                self.log_warning(
                    f"Condition check event {event} timed out after {event.timeout} ticks"
                )

        elif type(event) is AgentValidationEvent:
            val = event.get_validator()
            val._start_tick = self.tick_count  # type: ignore
            self.agent_action_validators.append(val)
        else:
            self.log_warning(f"Cannot process event type {type(event)}")

        # After processing events, we can now get handle the notifications
        # self.notification_system.handle_event(event)

        return successors  # type: ignore

    def resolve_arg_placeholders(
        self, event: Event, past_events: list[CompletedEvent]
    ) -> Event:
        """
        Resolve arg placeholders in the event.
        If an event has an arg that is a placeholder for a past event, we replace it with the return value of that event.
        """
        if event.action is None or event.action.args is None:
            return event

        for arg_name, arg_value in event.action.args.items():
            event.action.resolved_args[arg_name] = event.action.args[arg_name]

            if not isinstance(arg_value, str):
                continue

            match = re.match(r"^\{\{(.*?)\}\}$", arg_value.strip())

            if not match:
                continue

            parts = match.group(1).split(".")
            resolved_event_id = parts[0]

            for past_event in past_events:
                if resolved_event_id != past_event.event_id:
                    continue

                return_value = past_event.metadata.return_value
                resolved = True

                for key in parts[1:]:
                    if isinstance(return_value, dict) and key in return_value:
                        return_value = return_value[key]
                    else:
                        self.log_error(
                            f"Failed to find key {key} in return value of event {past_event.event_id}"
                        )
                        # Leave this argument at the placeholder it already holds in
                        # resolved_args and carry on with the remaining arguments.
                        # Returning here would leave them out of resolved_args, and
                        # Action.execute uses resolved_args whenever it is non-empty,
                        # so they would be dropped from the call entirely.
                        resolved = False
                        break

                if resolved:
                    event.action.resolved_args[arg_name] = return_value

                break

        return event

    def get_successors(self, event: AbstractEvent) -> list[AbstractEvent]:
        """
        Get the successors of an event.
        We also set the correct event time for the successors.
        """
        successors = []
        for successor in event.successors:
            if type(successor) not in {
                ConditionCheckEvent,
                Event,
                ValidationEvent,
                AgentValidationEvent,
                OracleEvent,
                StopEvent,
            }:
                raise ValueError(f"Invalid event type {type(successor)}")
            if successor.is_ready():
                successor.compute_absolute_time(
                    int(self.start_time) if self.start_time is not None else 0
                )
                self.log_debug(
                    f"Successor of {event.event_id}: Event {successor.event_id} scheduled at {datetime.fromtimestamp(successor.event_time, tz=timezone.utc)}"  # type: ignore
                )
                successors.append(successor)
        return successors

    def final_validation_checks(self):
        """
        Final validation checks to fail there are any remaining ValidationEvents or AgentValidationEvents
        """
        self.log_info("Performing final validation checks")
        for validator in self.agent_action_validators:
            if len(validator.milestones) > 0:
                raise ValidationException(
                    f"Agent Validator timed out, but {len(validator.milestones)} milestones are still not achieved: {validator.milestones}"
                )
        val_at_final_state = [
            e for e in self.event_queue.list_view() if type(e) is ValidationEvent
        ]
        if len(val_at_final_state) > 0:
            raise ValidationException(
                f"Validation events still active: {val_at_final_state}"
            )

    def register_apps(self, apps: list[App]):
        """
        Registers apps to the environment, this function initializes the add event to an app.
        It is also used to verify if a condition is possible to be met in the triggers
        """
        for app in apps:
            app.register_time_manager(self.time_manager)
            app.register_to_env("environment", self.add_to_log)
            if app.__class__ == AgentUserInterface:
                # Here for the AgentUserInterface we need to add the pause and resume functions
                # These are used to pause the event loop while waiting for user input and resume when done
                app.pause_env = self.pause
                app.resume_env = self.resume
            if isinstance(app, ReminderApp):
                # TODO: differentiate between Agent's reminder tooling and User's reminder app
                self.notification_system.setup_reminder_app(app)
            if isinstance(app, SystemApp):
                self.notification_system.setup_system_app(app)
                app.wait_for_next_notification = self.wait_for_next_notification
            for protocol in app.get_implemented_protocols():
                if protocol in self.protocol_to_app:
                    old_app = self.protocol_to_app[protocol].__class__.__name__
                    logging.warning(
                        f"Protocol {protocol} already registered by {old_app} app also provided by {app.__class__.__name__}"
                    )
                    continue
                self.protocol_to_app[protocol] = app
            self.apps[app.name] = app

        # connect apps to protocols once all protocols are gathered
        for app in self.apps.values():
            app.connect_to_protocols(self.protocol_to_app)

    def append_to_world_logs(self, world_log: BaseAgentLog):
        self.world_logs.append(world_log)

    def add_to_log(self, events: CompletedEvent | list[CompletedEvent]):
        """
        Add an event to the event log.
        """
        if not isinstance(events, list):
            events = [events]

        for event in events:
            self.event_log.put(event)

            if self.is_replaying:
                logger.info("Ignore this event since it happened during a replay")
                continue

            if event.event_type == EventType.AGENT:
                # Here we validate the action taken by an Agent before adding it to the event log
                try:
                    self.validate_agent_action(event)
                except ValidationException as e:
                    self.log_error(
                        f"Agent Action Validation failed on event {event.event_id} - with exception: {e}"
                    )
                    self.log_error("Stopping environment")
                    self.stop(final_state=EnvironmentState.FAILED)
                    return

            if self.notification_system is not None:
                self.notification_system.handle_event(event)

            action = event.action

            if not isinstance(action, Action):
                continue

            app_name = "unknown_app"

            if hasattr(action.app, "name"):
                app_name = action.app.name  # type: ignore
            elif action.app:
                app_name = action.app.app_name()

            content = get_content_for_message(event)

            action_log = ActionLog(
                content=(
                    content
                    if content is not None
                    else f"{event.app_class_name()}: {event.function_name()}"
                ),
                event_type=event.event_type.value,
                input={k: v for k, v in action.args.items() if k != "self"},
                output=event.metadata.return_value,
                action_name=action.function.__name__,
                app_name=app_name,
                exception=event.metadata.exception,
                exception_stack_trace=event.metadata.exception_stack_trace,
                timestamp=(
                    event.event_time
                    if event.event_time is not None
                    else self.env.time_manager.time()  # type: ignore[reportOptionalMemberAccess]
                ),
                agent_id="unknown",
            )

            action_log.id = event.event_id

            self.append_world_logs(action_log)

    def get_world_logs(self) -> list[BaseAgentLog]:
        # return a deep copy of the world logs
        return copy.deepcopy(self.world_logs)

    def set_world_logs(self, world_logs: list[BaseAgentLog]):
        self.world_logs = world_logs

    def append_world_logs(self, log: BaseAgentLog):
        self.world_logs.append(log)

    def increment_validators_tick_count(self):
        for validator in self.agent_action_validators:
            try:
                validator.update_tick_count(self.tick_count)  # type: ignore
            except ValidationException as e:
                self.log_error(f"Agent Action Validation failed with exception: {e}")
                self.log_error("Stopping environment")
                self.stop(final_state=EnvironmentState.FAILED)

    def validate_agent_action(self, event: CompletedEvent):
        """
        Validate an agent action.
        """
        for validator in self.agent_action_validators:
            # This is actually not great, converting a completed event to a future event
            # TODO are completed events even necessary ? We can just have a status field in the event like in the beginning
            validator.validate(self, event.to_future_event())

    def run(
        self,
        scenario: Scenario,
        wait_for_end: bool = True,
        schedule_events: bool = True,
    ):
        """
        Run the event loop
        """
        if not scenario._initialized:
            raise Exception(
                "Scenario not initialized, call scenario.initialize() first"
            )

        self.log_info(
            f"Running scenario {scenario.scenario_id} (duration={scenario.duration})"
        )

        self.time_manager.reset(start_time=self.start_time)
        self.duration = scenario.duration
        self.time_increment_in_seconds = scenario.time_increment_in_seconds
        self.delete_all_completed_events()
        self.register_apps(scenario.apps if scenario.apps else [])

        if schedule_events:
            self.schedule(scenario.events)

        if self.dump_dir is not None:
            self._dump_state("initial_state.jsonl")

        if AgentUserInterface.__name__ in self.apps:
            aui: AgentUserInterface = self.get_app(AgentUserInterface.__name__)  # type: ignore
            if self.environment_type == EnvironmentType.GUI:
                aui.set_cli(is_cli=False)
            elif self.environment_type == EnvironmentType.CLI:
                aui.set_cli(is_cli=True)

            if self.oracle_mode and aui.user_proxy is None:
                self.log_debug(
                    "Oracle mode is enabled but no user proxy is set, configuring AUI to not wait for user response"
                )
                aui.wait_for_user_response = False

        self.start()
        if wait_for_end:
            self.join()

    def schedule(
        self,
        events: AbstractEvent | list[AbstractEvent],
    ):
        """
        Add future event to the event queue.
        """
        events_to_schedule = self._get_events_to_schedule(events)
        for event in events_to_schedule:
            self.log_debug(
                f"Scheduling event {event.event_id} of type {type(event).__name__}"
            )

        self.events_to_schedule.extend(events_to_schedule)

    def _get_events_to_schedule(
        self, events: AbstractEvent | list[AbstractEvent]
    ) -> list[AbstractEvent]:
        """
        Get the events to schedule.
        """
        if not isinstance(events, list):
            events = [events]
        assert all([type(e) is not CompletedEvent for e in events]), (
            "Cannot schedule completed events"
        )
        events_to_schedule = events[:]
        return events_to_schedule

    def get_latest_event(self):
        """
        Get the last completed event that happened from the event log.
        """
        return self.event_log.past_events.peek()

    def get_past_event_at_index(self, index: int):
        """
        Get the past event at a specific index from the event log.
        WARNING: this function can be expensive as the log is stored in a priority queue for now.
        """
        return self.event_log.list_view()[index]

    def has_app(self, app_name: str) -> bool:
        """
        Check if an app is registered to the environment.
        """
        return app_name in self.apps

    def get_app(self, app_name: str) -> App:
        """
        Get an app by its name.
        """
        return self.apps[app_name]

    def get_app_with_class(self, app_class: Type[AppType]) -> AppType | None:
        """
        Check if an app is registered to the environment.
        """
        apps_with_class = [
            app for app in list(self.apps.values()) if isinstance(app, app_class)
        ]
        if len(apps_with_class) == 0:
            return None
        return apps_with_class[0]

    def get_tools_by_app(self) -> dict[str, list[AppTool]]:
        """
        Get for each app, the list of tools it has.
        """
        return {key: self.apps[key].get_tools() for key in self.apps.keys()}

    def get_tools(self) -> list[AppTool]:
        """
        Get the entire list of tools from all the apps.
        """
        return [
            item for sublist in self.get_tools_by_app().values() for item in sublist
        ]

    def get_user_tools_by_app(self) -> dict[str, list[AppTool]]:
        """
        Get for each app, the list of tools it has that are accessible to the User.
        """
        return {key: self.apps[key].get_user_tools() for key in self.apps.keys()}

    def get_user_tools(self) -> list[AppTool]:
        """
        Get the entire list of User tools from all the apps.
        """
        return [
            item
            for sublist in self.get_user_tools_by_app().values()
            for item in sublist
        ]

    def get_event_log_size(self) -> int:
        return len(self.event_log.list_view())

    def get_event_queue_length(self) -> int:
        return len(self.event_queue.list_view())

    def get_currently_active_validation_events(self) -> list[ValidationEvent]:
        """
        Get the currently active validation events.
        """
        return [
            e
            for e in self.event_queue.list_view()
            if type(e) is ValidationEvent
            and not e.is_timeout()
            and e.event_time <= self.current_time
        ]

    def _log(
        self, message: str, level: str, color: str, attrs: list[str] | None = None
    ):
        formatted_message = colored(
            f"[time = {datetime.fromtimestamp(self.current_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {message}",
            color,  # type: ignore
            attrs=attrs,  # type: ignore
        )

        if level == "info":
            logger.info(formatted_message)
        elif level == "debug":
            logger.debug(formatted_message)
        elif level == "warning":
            logger.warning(formatted_message)
        elif level == "error":
            logger.error(formatted_message)

    def log_info(self, message: str):
        self._log(message, "info", "blue")

    def log_debug(self, message: str):
        self._log(message, "debug", "light_grey")

    def log_warning(self, message: str):
        self._log(message, "warning", "yellow", ["bold"])

    def log_error(self, message: str):
        self._log(message, "error", "red", ["bold"])

    def is_paused(self) -> bool:
        return self.state == EnvironmentState.PAUSED

    def is_running(self) -> bool:
        return self.state == EnvironmentState.RUNNING
