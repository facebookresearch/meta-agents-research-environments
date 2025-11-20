# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from are.simulation.agents.multimodal import Attachment
from are.simulation.utils import make_serializable


@dataclass
class BaseAgentLog(ABC):
    timestamp: float
    id: str = field(init=False)
    agent_id: str

    def __post_init__(self) -> None:
        self.id = str(uuid.uuid4().hex)

    @abstractmethod
    def get_content_for_llm(self) -> str | None: ...

    def get_attachments_for_llm(self) -> list[Attachment]:
        """
        Contains attachments that should be sent to the LLM

        :return: Attachments to include in LLM message for this log entry.
        :rtype: list[Attachment]
        """
        return []

    @abstractmethod
    def get_type(self) -> str: ...

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["log_type"] = self.get_type()
        return data

    def serialize(self) -> str:
        return json.dumps(make_serializable(self.to_dict()))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseAgentLog":
        log_type = d.pop("log_type", None)
        id = d.pop("id", str(uuid.uuid4().hex))
        if log_type is None:
            raise ValueError("Log type is not specified")

        # Lazy import to avoid circular dependency
        from are.simulation.agents.custom_agents.browser_agent import BrowserStateLog

        log_type_map = {
            "system_prompt": SystemPromptLog,
            "task": TaskLog,
            "llm_input": LLMInputLog,
            "llm_output": LLMOutputThoughtActionLog,
            "llm_output_thought_action": LLMOutputThoughtActionLog,
            "rationale": RationaleLog,
            "tool_call": ToolCallLog,
            "observation": ObservationLog,
            "step": StepLog,
            "subagent": SubagentLog,
            "final_answer": FinalAnswerLog,
            "error": ErrorLog,
            "thought": ThoughtLog,
            "plan": PlanLog,
            "facts": FactsLog,
            "replan": ReplanLog,
            "refacts": RefactsLog,
            "stop": StopLog,
            "action": ActionLog,
            "end_task": EndTaskLog,
            "raw_plan": LLMOutputPlanLog,
            "llm_output_plan": LLMOutputPlanLog,
            "raw_facts": LLMOutputFactsLog,
            "llm_output_facts": LLMOutputFactsLog,
            "agent_user_interface": AgentUserInterfaceLog,
            "environment_notifications": EnvironmentNotificationLog,
            "hint": HintLog,
            "task_reminder": TaskReminderLog,
            "browser_state": BrowserStateLog,
        }

        log_class = log_type_map.get(log_type)
        if log_class is not None:
            log = log_class(**d)
            if log_type == "subagent":
                log.children = [
                    BaseAgentLog.from_dict(child) for child in d["children"]
                ]
            log.id = id
            return log

        raise ValueError(f"Unknown agent log type: {log_type}")


@dataclass
class SystemPromptLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "system_prompt"


@dataclass
class TaskLog(BaseAgentLog):
    content: str
    attachments: list[Attachment] = field(default_factory=list)

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_content_for_llm_no_attachment(self) -> str | None:
        content = re.sub(r"<\|attachment:(\d+)\|>", "", self.content)
        return content

    def get_attachments_for_llm(self) -> list[Attachment]:
        return self.attachments

    def get_type(self) -> str:
        return "task"


@dataclass
class LLMInputLog(BaseAgentLog):
    content: list[dict[str, str | list[Attachment]]]

    def get_content_for_llm(self) -> str | None:
        return None

    def get_type(self) -> str:
        return "llm_input"


@dataclass
class LLMOutputThoughtActionLog(BaseAgentLog):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    completion_duration: float = 0.0

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "llm_output"


@dataclass
class RationaleLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "rationale"


@dataclass
class ToolCallLog(BaseAgentLog):
    tool_name: str
    tool_arguments: str | dict[str, str]

    def get_content_for_llm(self) -> str | None:
        return json.dumps(
            {
                "tool_name": self.tool_name,
                "tool_arguments": self.tool_arguments,
            }
        )

    def get_type(self) -> str:
        return "tool_call"


@dataclass
class ObservationLog(BaseAgentLog):
    content: str
    attachments: list[Attachment] = field(default_factory=list)

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_content_for_llm_no_attachment(self) -> str | None:
        content = re.sub(r"<\|attachment:(\d+)\|>", "", self.content)
        return content

    def get_attachments_for_llm(self) -> list[Attachment]:
        return self.attachments

    def get_type(self) -> str:
        return "observation"


@dataclass
class StepLog(BaseAgentLog):
    iteration: int

    def get_content_for_llm(self) -> str | None:
        return None

    def get_type(self) -> str:
        return "step"


@dataclass
class SubagentLog(BaseAgentLog):
    group_id: str
    children: list[BaseAgentLog]
    name: str | None = None

    def get_content_for_llm(self) -> str | None:
        result = []

        for child in self.children:
            content_for_llm = child.get_content_for_llm()
            if content_for_llm is not None:
                result.append(content_for_llm)

        if len(result) == 0:
            return None
        else:
            return "\n".join(result)

    def get_type(self) -> str:
        return "subagent"

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["children"] = [child.to_dict() for child in self.children]
        return data


@dataclass
class FinalAnswerLog(BaseAgentLog):
    content: str
    attachments: list[Attachment] = field(default_factory=list)

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_content_for_llm_no_attachment(self) -> str | None:
        content = re.sub(r"<\|attachment:(\d+)\|>", "", self.content)
        return content

    def get_attachments_for_llm(self) -> list[Attachment]:
        return self.attachments

    def get_type(self) -> str:
        return "final_answer"


@dataclass
class ErrorLog(BaseAgentLog):
    error: str
    exception: str
    category: str
    agent: str

    def get_content_for_llm(self) -> str | None:
        return f"Error: {self.error}\nException: {self.exception}\nCategory: {self.category}"

    def get_type(self) -> str:
        return "error"


@dataclass
class ThoughtLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return None

    def get_type(self) -> str:
        return "thought"


@dataclass
class PlanLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "plan"


@dataclass
class FactsLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "facts"


@dataclass
class ReplanLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "replan"


@dataclass
class RefactsLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "refacts"


@dataclass
class StopLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return None

    def get_type(self) -> str:
        return "stop"


@dataclass
class ActionLog(BaseAgentLog):
    content: str
    input: dict[str, Any]
    event_type: str
    output: Any
    action_name: str
    app_name: str
    exception: str | None
    exception_stack_trace: str | None

    def get_content_for_llm(self) -> str | None:
        return f"Action: {self.action_name}\nInput: {self.input}\nOutput: {self.output}"

    def get_type(self) -> str:
        return "action"


@dataclass
class EndTaskLog(BaseAgentLog):
    def get_content_for_llm(self) -> str | None:
        return None

    def get_type(self) -> str:
        return "end_task"


@dataclass
class LLMOutputPlanLog(BaseAgentLog):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    completion_duration: float = 0.0

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "llm_output_plan"


@dataclass
class LLMOutputFactsLog(BaseAgentLog):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    completion_duration: float = 0.0

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "llm_output_facts"


@dataclass
class AgentUserInterfaceLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "agent_user_interface"


@dataclass
class EnvironmentNotificationLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "environment_notifications"


@dataclass
class HintLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "hint"


@dataclass
class TaskReminderLog(BaseAgentLog):
    content: str

    def get_content_for_llm(self) -> str | None:
        return self.content

    def get_type(self) -> str:
        return "task_reminder"
