# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import contextlib
import copy
import importlib
import inspect
import logging
import re
import threading
import traceback
import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Literal

# Optional strawberry import for GraphQL support
# When strawberry is not installed, we use no-op decorators
try:
    import strawberry

    HAS_STRAWBERRY = True
except ImportError:
    HAS_STRAWBERRY = False

    # Create no-op decorators when strawberry is not available
    class _StrawberryStub:
        def enum(self, cls):
            """No-op decorator when strawberry is not available"""
            return cls

        def type(self, cls):
            """Decorator that converts class to dataclass when strawberry is not available"""
            # Apply dataclass decorator to provide __init__ method
            return dataclass(cls)

    strawberry = _StrawberryStub()  # type: ignore

from are.simulation.priority_queue import PriorityQueue
from are.simulation.time_manager import TimeManager
from are.simulation.tool_utils import APPTOOL_ATTR_NAME, AppTool, OperationType
from are.simulation.utils import conditional_context_manager, get_function_name

if TYPE_CHECKING:
    from are.simulation.apps.app import App

logger = logging.getLogger(__name__)


@strawberry.enum
class EnvironmentState(Enum):
    """
    The state of the environment.
    - SETUP: the environment is being setup, this is the initial state of the environment
    - RUNNING: the environment event loop is running and events are being registered and logged
    - PAUSED: the environment is paused, and no events are being registered or logged, but can be restarted
    - STOPPED: the environment is completely stopped
    """

    SETUP = "SETUP"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    PAUSED = "PAUSED"
    FAILED = "FAILED"


@strawberry.enum
class EventTimeComparator(Enum):
    """
    Comparator for event time filtering.
    - LESS_THAN: Less than comparison
    - GREATER_THAN: Greater than comparison
    - EQUAL: Equal comparison
    """

    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    EQUAL = "EQUAL"


class AbstractEnvironment(ABC):
    state: EnvironmentState | None

    def __init__(self) -> None:
        super().__init__()
        self.time_increment_in_seconds = 0
        self.current_time = 0
        self.time_manager = TimeManager()
        self.start_time = 0
        self.event_log: EventLog = None  # type: ignore
        self.event_queue: EventQueue = None  # type: ignore

    def start(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError("Method is not yet implemented.")

    def get_app(self, app_name: str):
        raise NotImplementedError("Method is not yet implemented.")

    def get_event_log_size(self) -> int:
        raise NotImplementedError("Method is not yet implemented.")

    def get_event_queue_length(self) -> int:
        raise NotImplementedError("Method is not yet implemented.")

    def final_validation_checks(self) -> None:
        raise NotImplementedError("Method is not yet implemented.")


@strawberry.enum
class HintType(Enum):
    """
    Type of the hint, depends on the linked event
    - TASK_HINT: hints initiated by the send_message_to_agent
    """

    TASK_HINT = "TASK_HINT"
    ENVIRONMENT_HINT = "ENVIRONMENT_HINT"


@strawberry.type
@dataclass
class Hint:
    """
    Hint associated with an event
    - hint_type: Type of the hint, depends on the linked event
    - content: Content of the hint
    - associated_event_id: The id of the event that this hint is associated with
    """

    hint_type: HintType
    content: str
    associated_event_id: str


class EnvironmentType(Enum):
    """
    The type of the environment.
    """

    UNKNOWN = "UNKNOWN"
    CLI = "CLI"
    GUI = "GUI"


@strawberry.enum
class EventType(Enum):
    """
    Type of the events, depends on who initiated them.
    - AGENT: events initiated by the agent
    - ENV: events initiated by the environment, or the scenario designer.
    - USER: events initiated by the user or user proxy (unused for now).
    - CONDITION: events that check a condition and trigger other events.
    - VALIDATION: events that validate the state of the environment.
    - STOP: events that stop the simulation.
    """

    AGENT = "AGENT"
    ENV = "ENV"
    CONDITION = "CONDITION"
    VALIDATION = "VALIDATION"
    USER = "USER"
    STOP = "STOP"


@dataclass
class Action:
    """
    Action associated with an event, this is a function that will be called when the event is executed.

    - function: Function to be called when the action is executed, it can be a class method, or a regular function
    - args: Dict mapping the argument names to values to call the function with at execution time
    - app: The actual App instance to use for the action execution
           If not specified at creation time, it can be deducted from the function instance
           e.g. if function=email_app.add_emails then we can deduct that app=email_app
    - action_id: The unique id of the action, this is used to identify the event in the logs.
                 This is created automatically and does NOT need to be handled by the user
    - tool_metadata: Optional metadata from the AppTool that this action is associated with
    """

    function: Callable
    args: dict[str, Any] = field(default_factory=dict)
    resolved_args: dict[str, Any] = field(default_factory=dict)
    app: "App | None" = field(default=None)
    action_id: str = field(default=None)  # type: ignore
    operation_type: OperationType | None = field(default=OperationType.READ)
    tool_metadata: AppTool | None = field(default=None)

    def __post_init__(self):
        if self.action_id is None:
            self.action_id = f"{self.app.__class__.__name__}.{get_function_name(self.function)}-{uuid.uuid4()}"
        if self.app is None:
            if hasattr(self.function, "__self__"):
                # Import App here to avoid circular import
                from are.simulation.apps.app import App

                if issubclass(
                    self.function.__self__.__class__,  # type: ignore[reportFunctionMemberAccess]
                    App,
                ):
                    self.app = self.function.__self__  # type: ignore[reportFunctionMemberAccess]

        # Try to get the AppTool directly from the function
        apptool = AppTool.get_tool_for_function(self.function)
        if apptool is not None:
            self.tool_metadata = apptool

    def execute_on_app(self, app: "App"):
        self.app = app
        return self.execute()

    def execute(self):
        args = self.resolved_args if self.resolved_args else self.args
        if "self" in args:
            excluding_self = {k: v for k, v in args.items() if k != "self"}
            return self.function(self.app, **excluding_self)
        else:
            return self.function(**args)

    @property
    def function_name(self):
        return get_function_name(self.function)

    @property
    def class_name(self):
        return self.app.__class__.__name__

    @property
    def app_name(self):
        return self.app.name if self.app else self.class_name

    def __str__(self):
        filtered_args = {
            key: value for key, value in self.args.items() if key != "self"
        }
        return f"{self.class_name}.{self.function_name}({filtered_args})"

    def to_dict(self):
        result = {
            "class_name": self.class_name,
            "app_name": self.app_name,
            "function_name": self.function_name,
            "args": {key: value for key, value in self.args.items() if key != "self"},
            "resolved_args": {
                key: value for key, value in self.resolved_args.items() if key != "self"
            },
            "operation_type": self.operation_type.value,  # type: ignore[reportFunctionMemberAccess]
            "action_id": self.action_id,
        }

        # Include tool metadata if available
        if self.tool_metadata:
            result["tool_metadata"] = self.tool_metadata.to_metadata_dict()

        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        module = importlib.import_module("are.simulation.apps")
        class_from_module = getattr(module, d["class_name"])
        instance = class_from_module()
        method = getattr(instance, d["function_name"])
        action = cls(
            operation_type=OperationType(d["operation_type"].lower()),
            function=method,
            args=d["args"] if "args" in d else {},
            resolved_args=d["resolved_args"] if "resolved_args" in d else {},
            app=None,
            action_id=d["action_id"],
        )
        return action


@dataclass
class ActionDescription:
    app: str
    function: str
    args: list[dict[str, Any]]


@dataclass
class ConditionCheckAction:
    function: Callable[[AbstractEnvironment], bool]
    action_id: str = field(default=None)  # type: ignore

    def __post_init__(self):
        if self.action_id is None:
            self.action_id = f"{self.__class__.__name__}.{get_function_name(self.function)}-{uuid.uuid4()}"

    @property
    def function_name(self):
        return get_function_name(self.function)

    @property
    def class_name(self):
        return self.__class__.__name__

    def to_dict(self):
        return {
            "class_name": self.class_name,
            "function_name": self.function_name,
            "action_id": self.action_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            function=lambda env: True,
            action_id=d["action_id"],
        )


@dataclass
class EventMetadata:
    """
    Metadata for a completed event, which includes details such as the return value, exception, etc.
    This is particularly useful for completed events from App API calls, where:
    - return_value: the return value of the API call.
    - exception: any exception that occurred during the API call.
    - exception_stack_trace: the stack trace of the exception, if any.
    - completed: a flag indicating whether the execution was completed.
    """

    return_value: Any | None = None
    exception: str | None = None
    exception_stack_trace: str | None = None
    completed: bool = True

    def to_dict(self):
        return {
            "return_value": self.return_value,
            "exception": self.exception,
            "exception_stack_trace": self.exception_stack_trace,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EventMetadata":
        return cls(
            return_value=d.get("return_value"),
            exception=d.get("exception"),
            exception_stack_trace=d.get("exception_stack_trace"),
            completed=d.get("completed", True),
        )


@dataclass(order=True)
class AbstractEvent(ABC):
    """
    Abstract event class, that contains shared field between completed and future events.

    - event_type: the type of the event, either AGENT, ENV or USER.
    - action: the action that will be executed when the event happens, either directly a function or an Action obj.
    - event_time: the time at which the event will happen, this can get overridden in various placed for e.g. in case of conditional triggers.
    - event_relative_time: the relative time wrt the simulation start time
                           WARNING when the event is going to be added to the queue, this information is going to be used to set event_time
    - event_id: the unique id of the event, this is used to identify the event in the logs.
    """

    event_type: EventType = field(default=EventType.ENV)
    event_time: float | None = field(default=None)
    event_relative_time: float | None = field(default=None)
    event_id: str = field(default=None)  # type: ignore
    successors: list["AbstractEvent"] = field(default_factory=list)
    dependencies: list["AbstractEvent"] = field(default_factory=list)

    def __post_init__(self):
        if self.event_id is None:
            self.event_id = (
                f"{self.__class__.__name__}-{self.event_type.value}-{uuid.uuid4()}"
            )

    def __new__(cls, *args, **kwargs):
        if cls == AbstractEvent:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)

    def to_dict(self):
        return {
            "class_name": self.__class__.__name__,
            "event_type": self.event_type.value,
            "event_time": self.event_time,
            "event_relative_time": self.event_relative_time,
            "event_id": self.event_id,
            "successors": [s.to_dict() for s in self.successors],
            # Only showing dependencies ids to avoid infinite loops in serialization.
            "dependencies": [d.event_id for d in self.dependencies],
        }

    def depends_on(
        self,
        events: "AbstractEvent | list[AbstractEvent] | None" = None,
        delay_seconds: float = 0,
    ):
        """
        This function is used to add dependencies to the event.
        If e1 depends on e2 and e3, then e1 will only be executed after e2 and e3 are executed.
        If a delay is specified, then the event will be executed after the delay after the dependencies are executed.
        """
        assert delay_seconds >= 0, "Delay must be non-negative"

        self.event_relative_time = delay_seconds
        if events is None or (type(events) is list and len(events) == 0):
            return self

        if not isinstance(events, list):
            events = [events]
        for event in events:
            event.successors.append(self)
        self.dependencies.extend(events)
        return self

    def followed_by(
        self,
        events: "AbstractEvent | list[AbstractEvent]",
        delay_seconds: float | list[float] = 0.0,
    ):
        """
        This function is used to add successors to the event.
        If e1 is followed by e2 and e3, then e2 and e3 will only be executed after
        e1 is executed.
        If a delay is specified, then the event will be executed after the delay after the dependencies are executed.
        """
        if not isinstance(events, list):
            events = [events]
        if not isinstance(delay_seconds, list):
            delay_seconds = len(events) * [float(delay_seconds)]
        if len(events) != len(delay_seconds):
            raise ValueError("Number of events and delays must match")
        assert all(d >= 0 for d in delay_seconds), "Delay must be non-negative"
        for event, delay in zip(events, delay_seconds):
            event.event_relative_time = delay
            event.dependencies.append(self)
            self.successors.append(event)
        return self

    def is_ready(self) -> bool:
        """
        This function is used to check if the event is ready to be scheduled i.e. put into the event_queue.
        An event is ready to be executed if all its dependencies are executed.
        When an event has its event_time set, it means it is ready to be scheduled.
        """
        if self.event_time is not None:
            return True
        return (
            self.dependencies is None
            or len(self.dependencies) == 0
            or all(dep.event_time is not None for dep in self.dependencies)
        )

    def compute_absolute_time(self, start_time: int = 0):
        """
        Here we compute the absolute time of an event based on its relative time as well as the time of its dependencies.
        """
        if self.event_time is not None:
            # Skip calculation if absolute time is predefined
            return

        # Calculate the absolute time based on the maximum completion time of dependencies
        if len(self.dependencies) > 0:
            if any(dep.event_time is None for dep in self.dependencies):
                raise ValueError(
                    f"Cannot compute absolute time - Event {self.event_id} has dependencies that are not ready to be scheduled."
                )
            max_dependency_time = max(
                dep.event_time
                for dep in self.dependencies
                if dep.event_time is not None
            )
            self.event_time = max_dependency_time
            if self.event_relative_time is not None:
                self.event_time += self.event_relative_time  # type: ignore
        else:
            # No dependencies, schedule relative time from start
            self.event_time = start_time
            if self.event_relative_time is not None:
                self.event_time += self.event_relative_time  # type: ignore

    def delayed(self, delay: int):
        if delay >= 0:
            self.event_relative_time = delay
        return self

    def with_id(self, id: str):
        self.event_id = id
        return self

    def with_type(self, event_type: EventType):
        self.event_type = event_type
        return self

    def at_absolute_time(self, time: float):
        self.event_time = time
        return self

    def reset_dependencies(self):
        self.dependencies = []
        self.successors = []

    def copy(self):
        return copy.copy(self)


@dataclass(order=True)
class Event(AbstractEvent):
    """
    Represents an event that will happen in the future.
    This is what we create often when populating a scenario, and what gets added to the event queue.
    """

    action: Action = field(default=None)  # type: ignore

    def execute(self, fire_individual_events: bool = False) -> "CompletedEvent":
        """
        Executes the action corresponding to the events and returns the completed event with its metadata.
        Here by default we make sure we only have ONE event, and that whatever happens inside the action is not registered as individual events.
        This is to guarantee 2 things:
        1/ Events are transactional, either the whole Event (and associated action) happened or not
        2/ Not duplicating what is registered in the event log (otherwise we will register BOTH the current event, and every underlying one)
        """
        event_metadata = EventMetadata()
        try:
            with conditional_context_manager(
                not fire_individual_events, disable_events()
            ):
                return_value = self.action.execute()
                event_metadata.return_value = return_value
        except Exception as e:
            event_metadata.exception = str(e)
            event_metadata.exception_stack_trace = traceback.format_exc()

        return CompletedEvent(
            event_type=self.event_type,
            action=self.action,
            event_time=self.event_time,
            event_id=self.event_id,
            metadata=event_metadata,
        )

    def copy(self):
        return Event(
            event_type=self.event_type,
            action=self.action,
            event_time=self.event_time,
            event_id=self.event_id,
            dependencies=self.dependencies,
            successors=self.successors,
        )

    @classmethod
    def from_function(
        cls, function: Callable, event_type: EventType | None = None, **kwargs
    ):
        app_instance = None
        if isinstance(function, MethodType):
            # Import App here to avoid circular import
            from are.simulation.apps.app import App

            if isinstance(function.__self__, App):
                app_instance = function.__self__

        params: dict[str, Any] = dict(
            action=Action(
                app=app_instance,
                function=function,
                args=kwargs if kwargs else {},
            ),
        )
        if event_type is not None:
            params["event_type"] = event_type
        return Event(
            **params,
        )

    def app_class_name(self) -> str | None:
        if self.action.app is None:
            return None
        return self.action.app.__class__.__name__

    def app_name(self) -> str | None:
        if self.action.app is None:
            return None
        return self.action.app.name

    def function_name(self) -> str | None:
        if self.action.function is None:
            return None
        return get_function_name(self.action.function)

    def oracle(self):
        return OracleEvent.from_event(self)

    def to_dict(self):
        d = super().to_dict()
        d["class_name"] = self.__class__.__name__
        if type(self.action) is Action:
            d["action"] = self.action.to_dict()
        elif type(self.action) is ConditionCheckAction:
            d["action"] = self.action.to_dict()
        else:
            d["action"] = {}  # We don't support the validation action for now

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            event_type=EventType(d["event_type"]),
            event_time=d["event_time"],
            event_relative_time=d.get("event_relative_time", None),
            event_id=d["event_id"],
            action=Action.from_dict(d["action"]),
            dependencies=d["dependencies"],
        )


class StopEvent(AbstractEvent):
    def __init__(self):
        super().__init__(
            event_type=EventType.STOP,
        )

    def execute(self, fire_individual_events: bool = False) -> "CompletedEvent":
        return CompletedEvent(
            event_type=self.event_type,
            event_time=self.event_time,
            event_id=self.event_id,
            metadata=EventMetadata(),
        )


@dataclass(order=True)
class CompletedEvent(AbstractEvent):
    """
    Represents an event that already happened, and thus we have some additional metadata on it.
    """

    action: Action | ConditionCheckAction = field(default=None)  # type: ignore
    metadata: EventMetadata = field(default=None)  # type: ignore
    _tool_name: str | None = field(default=None)

    def app_class_name(self) -> str | None:
        if type(self.action) is ConditionCheckAction or self.action.app is None:  # type: ignore
            return None
        return self.action.app.__class__.__name__  # type: ignore

    def app_name(self) -> str | None:
        if type(self.action) is ConditionCheckAction or self.action.app is None:  # type: ignore
            return None
        return self.action.app.name  # type: ignore

    def function_name(self) -> str | None:
        if self.action.function is None:
            return None
        return get_function_name(self.action.function)

    def replay(self):
        if type(self.action) is Action and self.action.app is not None:  # type: ignore
            self.action.execute()

    def failed(self) -> bool:
        return self.metadata.exception is not None

    def copy(self):
        return CompletedEvent(
            event_type=self.event_type,
            action=self.action,
            event_time=self.event_time,
            event_id=self.event_id,
            metadata=self.metadata,
        )

    def to_future_event(self):
        if type(self.action) is Action:
            return Event(
                action=self.action,  # type: ignore
                event_type=self.event_type,
                event_time=self.event_time,
                event_id=self.event_id,
            )
        elif type(self.action) is ConditionCheckAction:
            return ConditionCheckEvent(
                action=self.action,  # type: ignore
                event_type=self.event_type,
                event_time=self.event_time,
                event_id=self.event_id,
            )
        else:
            raise ValueError(f"Action {self.action} not supported")

    def to_dict(self):
        d = super().to_dict()
        d["class_name"] = self.__class__.__name__
        d["metadata"] = self.metadata.to_dict()
        if type(self.action) is Action:
            d["action"] = self.action.to_dict()
        elif type(self.action) is ConditionCheckAction:
            d["action"] = self.action.to_dict()
        else:
            d["action"] = {}  # We don't support the validation action for now

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            event_type=EventType(d["event_type"]),
            event_time=d["event_time"],
            event_relative_time=d.get("event_relative_time", None),
            event_id=d["event_id"],
            action=(
                ConditionCheckAction.from_dict(d["action"])
                if d["action"]["class_name"] == "ConditionCheckAction"
                else Action.from_dict(d["action"])
            ),
            metadata=EventMetadata.from_dict(d["metadata"]),
            dependencies=d["dependencies"],
        )

    @property
    def tool_name(self):
        """Tool name used for validation"""
        if self._tool_name is not None:
            # If the tool name is overwritten
            return self._tool_name
        app_class_name = self.app_class_name()
        app_class_name = app_class_name if app_class_name else "NoApp"
        fn_name = self.function_name()
        fn_name = fn_name if fn_name else "NoFunction"
        return f"{app_class_name}__{fn_name}"

    def get_args(self) -> dict[str, Any]:
        if isinstance(self.action, ConditionCheckAction):
            return {}
        return (
            self.action.resolved_args if self.action.resolved_args else self.action.args
        )


@dataclass(order=True)
class ConditionCheckEvent(AbstractEvent):
    event_type: EventType = field(default=EventType.CONDITION)
    action: ConditionCheckAction = field(default=None)  # type: ignore
    schedule_every_ticks: int = field(default=1)
    timeout: int | None = field(default=None)
    _internal_check_count: int = field(default=0)

    def __post_init__(self):
        super().__post_init__()
        self._add_check_count_to_id()

    def with_id(self, id: str):
        self.event_id = id
        self._add_check_count_to_id()
        return self

    def _add_check_count_to_id(self):
        match = re.search(r"CHECK_(\d+)", self.event_id)
        if match:
            updated_id = re.sub(
                r"CHECK_\d+", f"CHECK_{self._internal_check_count}", self.event_id
            )
            self.event_id = updated_id
        else:
            self.event_id += f"-CHECK_{self._internal_check_count}"

    def is_timeout(self) -> bool:
        if self.timeout is None:
            return False
        return self._internal_check_count * self.schedule_every_ticks > self.timeout

    @classmethod
    def from_condition(
        cls,
        condition: Callable[[AbstractEnvironment], bool],
        every_tick: int = 1,
        timeout: int | None = None,
    ):
        return ConditionCheckEvent(
            action=ConditionCheckAction(function=condition),
            schedule_every_ticks=every_tick,
            timeout=timeout,
        )

    def copy(self):
        return ConditionCheckEvent(
            event_type=self.event_type,
            action=self.action,
            event_time=self.event_time,
            event_id=self.event_id,
            dependencies=self.dependencies,
            successors=self.successors,
            schedule_every_ticks=self.schedule_every_ticks,
            timeout=self.timeout,
            _internal_check_count=self._internal_check_count,
        )

    def check(self, env: AbstractEnvironment) -> tuple[bool, CompletedEvent]:
        self._internal_check_count += 1
        success = self.action.function(env)
        completed_check = CompletedEvent(
            event_type=self.event_type,
            action=self.action,
            event_time=self.event_time,
            event_id=self.event_id,
            metadata=EventMetadata(
                return_value=success,
            ),
        )

        return success, completed_check

    def depends_on(
        self,
        events: AbstractEvent | list[AbstractEvent] | None = None,
        delay_seconds: float = 0,
        schedule_every_ticks: int | None = None,
        timeout: int | None = None,
    ):
        if delay_seconds < 0:
            raise ValueError("Delay must be non-negative")
        if schedule_every_ticks is not None:
            schedule_every_ticks = self.schedule_every_ticks
        if timeout is not None:
            self.timeout = timeout
        self.event_relative_time = delay_seconds
        if events is None:
            return self
        if not isinstance(events, list):
            events = [events]
        for event in events:
            event.successors.append(self)
        self.dependencies.extend(events)
        return self

    def get_next_check_event(self, time_increment_in_seconds: int):
        new_condition_check = ConditionCheckEvent(
            action=self.action,
            event_time=self.event_time  # type: ignore
            + self.schedule_every_ticks * time_increment_in_seconds,
            event_id=self.event_id,
            dependencies=self.dependencies,
            successors=self.successors,
            schedule_every_ticks=self.schedule_every_ticks,
            timeout=self.timeout,
            _internal_check_count=self._internal_check_count,
        )
        # We need to replace the current event with the new one in the dependencies of the successors
        # otherwise if reference is kept to the old condition check, scheduling time will be wrong
        for successor in self.successors:
            successor.dependencies.remove(self)
            successor.dependencies.append(new_condition_check)
        return new_condition_check


@dataclass
class OracleEvent(AbstractEvent):
    make_event: Callable[[AbstractEnvironment], AbstractEvent] = field(default=None)  # type: ignore
    event_type: EventType = field(default=EventType.AGENT)
    event_time_comparator: EventTimeComparator | None = field(default=None)
    action_desc: ActionDescription | None = field(default=None)

    def make(self, env: AbstractEnvironment):
        with EventRegisterer.capture_mode():
            event = (
                self.make_event(env).with_type(self.event_type).with_id(self.event_id)
            )
            event.event_relative_time = self.event_relative_time
            event.event_time = self.event_time
            event.dependencies = self.dependencies
            event.successors = self.successors
        return event

    @classmethod
    def from_event(cls, event: AbstractEvent):
        return OracleEvent(
            make_event=lambda env: event,
            event_time=event.event_time,
            event_relative_time=event.event_relative_time,
            successors=event.successors,
            dependencies=event.dependencies,
            action_desc=cls.action_desc_from_event(event),
            event_id=event.event_id,
        )

    def to_dict(self):
        d = super().to_dict()
        d["event_time_comparator"] = self.event_time_comparator
        if self.action_desc:
            d["action"] = {
                "class_name": self.action_desc.app,
                "app_name": self.action_desc.app,
                "function_name": self.action_desc.function,
                "args": {v["name"]: v["value"] for v in self.action_desc.args},
            }
        return d

    @classmethod
    def action_desc_from_event(cls, event: AbstractEvent):
        if isinstance(event, Event):
            action: Action = event.action
            return ActionDescription(
                app=action.app_name,
                function=action.function_name,
                args=[
                    {"name": k, "value": str(v), "value_type": type(v).__name__}
                    for k, v in action.args.items()
                    if k != "self"
                ],
            )
        else:
            return None


@dataclass
class CompletedOracleEvent(CompletedEvent):
    """
    A completed oracle event with timing information from the original oracle event.
    """

    absolute_event_time: float | None = None
    event_time_comparator: EventTimeComparator | None = None

    @classmethod
    def from_completed_event(
        cls,
        completed_event: CompletedEvent,
        absolute_event_time: float | None = None,
        event_time_comparator: EventTimeComparator | None = None,
    ):
        return cls(
            absolute_event_time=absolute_event_time,
            event_time_comparator=event_time_comparator,
            **completed_event.to_dict(),
        )

    @classmethod
    def from_completed_event_and_oracle_event(
        cls, completed_event: CompletedEvent, oracle_event: AbstractEvent
    ):
        """
        Create a completed oracle event from a completed event and an oracle event.
        The absolute event time is taken from the oracle event, and the event time comparator is taken from the oracle event if it exists.
        """
        if not isinstance(oracle_event, OracleEvent) and not isinstance(
            oracle_event, Event
        ):
            raise ValueError(
                f"oracle_event must be an instance of OracleEvent or Event, not {type(oracle_event)}"
            )
        return cls(
            absolute_event_time=oracle_event.event_time,
            event_time_comparator=(
                oracle_event.event_time_comparator
                if type(oracle_event) is OracleEvent
                else None
            ),
            **completed_event.__dict__,
        )


@dataclass
class ValidationResult:
    """
    Represents the result of a validation event.
    - success: whether the validation was successful or not.
    - message: a message describing the result of the validation.
    - failed_milestones: the list of milestones that failed during the validation.
    - triggered_minefields: the list of minefields that were triggered during the validation.
    """

    success: bool
    achieved_milestones: (
        list[Callable[[AbstractEnvironment], bool]]
        | list[Callable[[AbstractEnvironment, AbstractEvent], bool]]
    ) = field(default_factory=list)

    def __str__(self):
        return f"ValidationResult(success={self.success}, achieved_milestones={self.achieved_milestones})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "success": self.success,
            "message": self.message,  # type: ignore
            "achieved_milestones": [str(m) for m in self.achieved_milestones],
        }


@dataclass
class AgentActionValidator:
    milestones: list[Callable[[AbstractEnvironment, AbstractEvent], bool]] = field(
        default_factory=list
    )
    minefields: list[Callable[[AbstractEnvironment, AbstractEvent], bool]] = field(
        default_factory=list
    )
    timeout: int | None = field(default=None)
    _start_tick: int = field(default=0)
    _internal_check_count: int = field(default=0)
    achieved_milestones: list[Callable[[AbstractEnvironment, AbstractEvent], bool]] = (
        field(default_factory=list)
    )

    def is_timeout(self) -> bool:
        if self.timeout is None:
            return False
        return self._internal_check_count >= self.timeout

    def update_tick_count(self, current_env_tick: int):
        self._internal_check_count = current_env_tick - self._start_tick

    def validate(
        self, env: AbstractEnvironment, event: AbstractEvent
    ) -> ValidationResult:
        # need to check timeout first, otherwise if the milestone is achieved in validate(), it will be removed from the list and the validation will pass
        if self.is_timeout() and len(self.milestones) > 0:
            raise ValidationException(
                f"Agent Validation timed out, but {len(self.milestones)} milestones are still not achieved: {self.milestones}"
            )
        if self.is_timeout() and len(self.minefields) > 0:
            self.minefields = []

        milestones_to_remove = []
        for milestone in self.milestones:
            # Once a milestone is achieved, we remove it from the list of milestones to check
            if milestone(env, event):
                self.achieved_milestones.append(milestone)
                milestones_to_remove.append(milestone)

        for m in milestones_to_remove:
            self.milestones.remove(m)

        # Check if any minefield is triggered
        triggered_minefields = []
        for minefield in self.minefields:
            if minefield(env, event):
                triggered_minefields.append(minefield)

        # If any minefield is triggered, the validation fails immediately
        if len(triggered_minefields) > 0:
            raise ValidationException(
                f"Agent event {event.event_id} triggered {len(triggered_minefields)} minefields: {triggered_minefields}"
            )

        validation_result = ValidationResult(
            success=(
                len(self.achieved_milestones) == len(self.milestones)
            ),  # Validation is successful if all milestones are achieved
            achieved_milestones=self.achieved_milestones[:],  # make a copy here
        )

        return validation_result


@dataclass
class AgentValidationEvent(AbstractEvent):
    event_type: EventType = field(default=EventType.VALIDATION)
    validators: list[AgentActionValidator] = field(default_factory=list)
    milestones: list[Callable[[AbstractEnvironment, AbstractEvent], bool]] = field(
        default_factory=list
    )
    minefields: list[Callable[[AbstractEnvironment, AbstractEvent], bool]] = field(
        default_factory=list
    )
    timeout: int | None = field(default=None)

    def get_validator(self):
        return AgentActionValidator(
            milestones=self.milestones[:],
            minefields=self.minefields[:],
            timeout=self.timeout,
        )

    def depends_on(
        self,
        events: AbstractEvent | list[AbstractEvent] | None = None,
        delay_seconds: float = 0,
        schedule_every_ticks: int | None = None,
        timeout: int | None = None,
    ):
        if delay_seconds < 0:
            raise ValueError("Delay must be non-negative")
        if schedule_every_ticks is not None:
            self.schedule_every_ticks = schedule_every_ticks
        if timeout is not None:
            self.timeout = timeout
        self.event_relative_time = delay_seconds
        if events is None:
            return self
        if not isinstance(events, list):
            events = [events]
        for event in events:
            event.successors.append(self)
        self.dependencies.extend(events)
        return self

    def schedule(self, every_ticks: int, timeout: int | None = None):
        self.schedule_every_ticks = every_ticks
        self.timeout = timeout
        return self


class ValidationException(Exception):
    pass


@dataclass(order=True)
class ValidationEvent(AbstractEvent):
    event_type: EventType = field(default=EventType.VALIDATION)
    # Milestones are conditions that absolutely need to be achieved for the validation to be successful
    milestones: list[Callable[[AbstractEnvironment], bool]] = field(
        default_factory=list
    )
    # Minefields are conditions that if triggered, the validation will fail immediately
    minefields: list[Callable[[AbstractEnvironment], bool]] = field(
        default_factory=list
    )
    schedule_every_ticks: int = field(default=1)
    timeout: int | None = field(default=1)
    _internal_check_count: int = field(default=0)
    achieved_milestones: list[Callable[[AbstractEnvironment], bool]] = field(
        default_factory=list
    )

    def is_timeout(self) -> bool:
        if self.timeout is None:
            return False
        return self._internal_check_count * self.schedule_every_ticks >= self.timeout

    def validate(
        self, env: AbstractEnvironment
    ) -> tuple[ValidationResult, CompletedEvent]:
        self._internal_check_count += 1

        milestones_to_remove = []
        for milestone in self.milestones:
            # Once a milestone is achieved, we remove it from the list of milestones to check
            if milestone(env):
                self.achieved_milestones.append(milestone)
                milestones_to_remove.append(milestone)

        for m in milestones_to_remove:
            self.milestones.remove(m)

        # Check if any minefield is triggered
        triggered_minefields = []
        for minefield in self.minefields:
            if minefield(env):
                triggered_minefields.append(minefield)

        # If any minefield is triggered, the validation fails immediately
        if len(triggered_minefields) > 0:
            raise ValidationException(
                f"Validation event {self.event_id} triggered {len(triggered_minefields)} minefields: {triggered_minefields}"
            )

        validation_result = ValidationResult(
            success=len(self.milestones)
            == 0,  # Validation is successful if all milestones are achieved
            achieved_milestones=self.achieved_milestones[:],  # make a copy here
        )

        completed_event = CompletedEvent(
            event_type=self.event_type,
            event_time=self.event_time,
            event_id=self.event_id,
            action=self.validate,  # type: ignore
            metadata=EventMetadata(return_value=validation_result),
        )

        return validation_result, completed_event

    def get_next_event(
        self, time_increment_in_seconds: int = 1
    ) -> "ValidationEvent | None":
        # Get the next validation event to be scheduled depending on the milestones achieved
        if len(self.milestones) == 0 and len(self.minefields) == 0:
            # If all of them are achieved, and we don't need to check for any minefield, then we return None
            # Which means no further validation events are needed
            return None
        elif self.is_timeout():
            # If Validation is already timed out while some milestones are still not achieved, then we raise an exception
            if len(self.milestones) > 0:
                raise ValidationException(
                    f"Validation event {self.event_id} timed out, but {len(self.milestones)} milestones are still not achieved: {self.milestones}"
                )
            else:
                # otherwise it means validation is done ! and nothing further to schedule
                return None

        new_validation_event = ValidationEvent(
            event_type=self.event_type,
            event_time=self.event_time  # type: ignore
            + self.schedule_every_ticks * time_increment_in_seconds,
            event_id=self.event_id,
            milestones=self.milestones[:],
            minefields=self.minefields[:],
            schedule_every_ticks=self.schedule_every_ticks,
            timeout=self.timeout,
            _internal_check_count=self._internal_check_count,
            achieved_milestones=self.achieved_milestones[:],
        )

        # We need to replace the current event with the new one in the dependencies of the successors
        # otherwise if reference is kept to the old condition check, scheduling time will be wrong
        for successor in self.successors:
            successor.dependencies.remove(self)
            successor.dependencies.append(new_validation_event)
        return new_validation_event

    def copy(self):
        return ValidationEvent(
            event_type=self.event_type,
            event_time=self.event_time,
            event_id=self.event_id,
            dependencies=self.dependencies,
            successors=self.successors,
            milestones=self.milestones[:],
            minefields=self.minefields[:],
            schedule_every_ticks=self.schedule_every_ticks,
            timeout=self.timeout,
            _internal_check_count=self._internal_check_count,
            achieved_milestones=self.achieved_milestones[:],
        )

    def depends_on(
        self,
        events: AbstractEvent | list[AbstractEvent] | None = None,
        delay_seconds: float = 0,
        schedule_every_ticks: int | None = None,
        timeout: int | None = None,
    ):
        if delay_seconds < 0:
            raise ValueError("Delay must be non-negative")
        if schedule_every_ticks is not None:
            schedule_every_ticks = self.schedule_every_ticks
        if timeout is not None:
            self.timeout = timeout
        self.event_relative_time = delay_seconds
        if events is None:
            return self
        if not isinstance(events, list):
            events = [events]
        for event in events:
            event.successors.append(self)
        self.dependencies.extend(events)
        return self

    def schedule(self, every_ticks: int, timeout: int | None = None):
        self.schedule_every_ticks = every_ticks
        self.timeout = timeout
        return self


@dataclass
class EventLog:
    """
    Event log, contains all the events that happened so far in the environment.
    """

    past_events: PriorityQueue[CompletedEvent] = field(
        default_factory=lambda: PriorityQueue[CompletedEvent](fields=["event_time"])
    )

    def put(self, event: CompletedEvent | list[CompletedEvent]):
        if not isinstance(event, list):
            event = [event]
        for event in event:
            # We copy the event here to avoid the event logged to be mutated later
            event_copy = event.copy()
            self.past_events.put(event_copy)

    def __len__(self):
        return self.past_events.qsize()

    def list_view(self) -> list[CompletedEvent]:
        return list(self.past_events)

    def to_dict(self):
        return {
            "past_events": [event.to_dict() for event in self.list_view()],
        }

    @staticmethod
    def from_list_view(events: list[CompletedEvent]):
        event_log = EventLog()
        for event in events:
            event_log.past_events.put(event)
        return event_log


@dataclass
class EventQueue:
    """
    Event queue, contains all the events that will happen in the future.
    """

    future_events: PriorityQueue[AbstractEvent] = field(
        default_factory=lambda: PriorityQueue[AbstractEvent](
            fields=["event_time", "event_id"]
        )
    )
    already_scheduled: set[str] = field(default_factory=set)

    def put(
        self,
        events: Event | ConditionCheckEvent | list[Event | ConditionCheckEvent],
    ):
        if not isinstance(events, list):
            events = [events]

        for event in events:
            # We copy the event here to avoid the event in the queue to be mutated later
            # We also avoid scheduling multiple times the same Event instance
            # but for ConditionalCheckEvent it is ok to schedule multiple times
            if (
                isinstance(event, (Event, OracleEvent))
                and event.event_id in self.already_scheduled
            ):
                logger.debug(f"Event {event.event_id} already scheduled, skipping")
                continue

            self.future_events.put(event)
            self.already_scheduled.add(event.event_id)

    def pop_events_to_process(self, timestamp: float):
        extracted_events = []
        remaining_events = []

        while not self.future_events.empty():
            event = self.future_events.get()
            if event.event_time <= timestamp:
                extracted_events.append(event)
            else:
                remaining_events.append(event)
                break  # since events are ordered, no need to check further

        # Reinsert the remaining items back into the queue
        for item in remaining_events:
            self.future_events.put(item)

        return extracted_events

    def __len__(self):
        return self.future_events.qsize()

    def peek(self):
        return self.future_events.peek()

    def list_view(self) -> list[Event]:
        return list(self.future_events)

    def to_dict(self):
        return {
            "future_events": [event.to_dict() for event in self.list_view()],
        }

    def from_list_view(self, events: list[AbstractEvent]):
        event_queue = EventQueue()
        for event in events:
            event_queue.future_events.put(event)
        return event_queue


class EventRegisterer:
    """
    Class that handles all the logic for registering events.
    """

    # We make this variable thread local, so that every thread can disable and enable event firing without affecting other threads
    _thread_local = threading.local()

    @classmethod
    def is_active(cls):
        """
        Checks whether we should fire events or not.
        """
        return getattr(cls._thread_local, "active", True)

    @classmethod
    def is_capture_mode(cls):
        """
        Capture mode makes sure a function call is not executed but only a "fictitious" CompletedEvent is returned.
        This is useful for debugging and testing, as well as easily creating CompletedEvent instances for validation
        """
        return getattr(cls._thread_local, "capture_mode", False)

    @classmethod
    def set_active(cls, state):
        """
        Sets whether we should fire events or not.
        """
        cls._thread_local.active = state

    @classmethod
    def set_capture_mode(cls, state):
        """
        Sets whether capture mode is active or not.
        """
        cls._thread_local.capture_mode = state

    @classmethod
    def event_registered(
        cls,
        operation_type: OperationType = OperationType.READ,
        event_type: EventType = EventType.AGENT,
    ):
        """
        This decorator is used to wrap API calls, so that we can register the event and add the appropriate CompletedEvent instance.
        The CompletedEvent instance is only added to the Event Log if the App is already registered in the environment.

        This decorator is also used to capture fictitious CompletedEvent instances when capture mode is active.
        Capture mode allows to easily simulate and create CompletedEvent instances without actually executing the API call.
        This is useful for debugging and testing, as well as defining validation trajectories.
        """

        def with_event(func: Callable) -> Callable:
            func.__event_registered__ = True  # type: ignore
            func.__operation_type__ = operation_type  # type: ignore

            @wraps(func)
            def wrapper(self, *args, **kwargs) -> Any:
                # We only apply the event building and registering logic if active, otherwise we will just call the function normally.
                if not cls.is_active():
                    return func(self, *args, **kwargs)
                else:
                    action_id = f"{self.name}.{func.__name__}-{uuid.uuid4()}"
                    bound_arguments = inspect.signature(func).bind(
                        self, *args, **kwargs
                    )
                    bound_arguments.apply_defaults()
                    func_args = bound_arguments.arguments
                    action = Action(
                        app=self,
                        function=func,
                        args=func_args,
                        operation_type=operation_type,
                    )

                    if cls.is_capture_mode():
                        # We are in capture mode, so we just return an Event here, without executing anything
                        return Event(
                            event_id=f"{EventType.ENV.value}-{action_id}",
                            event_type=EventType.ENV,
                            action=action,
                        )
                    else:
                        event_metadata = EventMetadata()
                        event_time = self.time_manager.time()
                        # We are not in capture mode, so we execute the action and return the result
                        try:
                            result = func(self, *args, **kwargs)
                            event_metadata.return_value = result
                        except Exception as e:
                            event_metadata.exception = str(e)
                            event_metadata.exception_stack_trace = (
                                traceback.format_exc()
                            )
                            raise e
                        finally:
                            event = CompletedEvent(
                                event_id=f"{event_type.value}-{action_id}",
                                event_type=event_type,
                                action=action,
                                metadata=event_metadata,
                                event_time=event_time,
                            )
                            self.add_event(event)
                return result

            # Propagate the AppTool metadata between the original function and the wrapper function
            # Check if the function has the AppTool attribute
            apptool = getattr(func, APPTOOL_ATTR_NAME, None)
            if apptool is not None:
                setattr(wrapper, APPTOOL_ATTR_NAME, apptool)

            # Add a function to set the AppTool metadata on both the wrapper and the original function
            def set_apptool(app_tool_instance):
                setattr(wrapper, APPTOOL_ATTR_NAME, app_tool_instance)
                setattr(func, APPTOOL_ATTR_NAME, app_tool_instance)

            # Attach the set_apptool function to the wrapper
            wrapper.set_apptool = set_apptool  # type: ignore

            return wrapper

        return with_event

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        """
        Context manager to disable event firing, and just return the result of the function call as if decorator is not applied.
        """
        original_state = cls.is_active()
        cls.set_active(False)
        try:
            yield
        finally:
            cls.set_active(original_state)

    @classmethod
    @contextlib.contextmanager
    def capture_mode(cls):
        """
        Context manager to replace a function call decorated with event_registered, by just a fictitious CompletedEvent instance.
        """
        original_state = cls.is_capture_mode()
        cls.set_capture_mode(True)
        try:
            yield
        finally:
            cls.set_capture_mode(original_state)


event_registered = EventRegisterer.event_registered
disable_events = EventRegisterer.disable


@strawberry.enum
class CapabilityTag(Enum):
    Planning = "Planning"
    Memory = "Memory"
    Collaboration = "Collaboration"
    Exploration = "Exploration"
    UnitTest = "UnitTest"
    PromptInjection = "PromptInjection"
    Universe = "Universe"
    Safety = "Safety"
    # Gaia V2 capabilities
    Adaptability = "Adaptability"
    Ambiguity = "Ambiguity"
    Execution = "Execution"
    Search = "Search"  # Replace DeepSearch
    Time = "Time"  # Replace ProspectiveMemory
    Security = "Security"

    @classmethod
    def gaia2_capabilities(cls):
        return [cls.Ambiguity, cls.Adaptability, cls.Execution, cls.Search, cls.Time]


@dataclass
class ScenarioGUIConfig:
    show_timestamps: bool = False


@dataclass
class ToolAugmentationConfig:
    tool_failure_probability: float = 0.1
    # Mapping from original tool name to the new tool name
    apply_tool_name_augmentation: bool = True
    apply_tool_description_augmentation: bool = True


@dataclass
class SimulatedGenerationTimeConfig:
    """Configuration for simulating the LLM's generation time in are.simulation.

    Modes:
        - fixed: The simulated generation time is set to a fixed to `seconds`.
        - measured: The simulated generation time is measured from the first successful generation.
    """

    mode: Literal["fixed", "measured"] = "measured"
    seconds: float | None = 1.0  # Used when mode is "fixed" and as mean for "random"

    def __post_init__(self):
        if self.mode == "fixed":
            if self.seconds is None:
                raise ValueError(
                    f"When mode is '{self.mode}', seconds must be provided and cannot be None."
                )


@dataclass
class PlaceholderMetadata:
    """
    Metadata for a placeholder.
    """

    parent_tool_name: str  # Name of the tool parent event.
    parent_turn_idx: int  # Index of the turn the parent event belongs to.
    parent_event_id: str  # ID of the parent event.
    placeholder_turn_idx: int  # Index of the turn the placeholder belongs to.
    placeholder_event_id: str  # ID of the placeholder event.


@dataclass
class ExecutionMetadata:
    """
    Data structure for execution data.
    """

    has_placeholder_conflicts: bool
    placeholders: list[PlaceholderMetadata]
