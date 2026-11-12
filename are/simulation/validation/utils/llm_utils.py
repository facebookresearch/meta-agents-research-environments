# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from are.simulation.scenarios.utils.personalization.utils import jinja_format
from are.simulation.validation.constants import SoftCheckerType
from are.simulation.validation.prompts import (
    CAB_CHECKER_PROMPT_TEMPLATES,
    CONTENT_CHECKER_PROMPT_TEMPLATES,
    EMAIL_CHECKER_PROMPT_TEMPLATES,
    EVENT_CHECKER_PROMPT_TEMPLATES,
    EVENT_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
    MESSAGE_CHECKER_PROMPT_TEMPLATES,
    SANITY_CHECKER_PROMPT_TEMPLATES,
    SIGNATURE_CHECKER_TEMPLATES,
    SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
    TONE_CHECKER_PROMPT_TEMPLATES,
    USER_MESSAGE_CHECKER_PROMPT_TEMPLATES,
    USER_MESSAGE_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
    LLMFunctionTemplates,
)

logger = logging.getLogger(__name__)


class LLMFunctionBase(ABC):
    """
    LLMFunctionBase is a base class for facilitating interaction with a language model engine.
    """

    def __init__(
        self, engine: Callable, prompt_templates: LLMFunctionTemplates
    ) -> None:
        self.engine = engine
        self.prompt_templates = prompt_templates

    @abstractmethod
    def __call__(self, user_prompt_args: dict[str, str]) -> Any:
        pass


class LLMFunction(LLMFunctionBase):
    """
    LLMFunction is a class that extends LLMFunctionBase to support system, user, and assistant prompts,
    and can include examples for context. The __call__ method sends the constructed messages to the engine
    and returns the response.
    """

    def __init__(
        self, engine: Callable, prompt_templates: "LLMFunctionTemplates"
    ) -> None:
        super().__init__(engine, prompt_templates)
        # System prompt template
        system_prompt_args = prompt_templates.system_prompt_args or {}
        self.system_prompt = jinja_format(
            template=prompt_templates.system_prompt_template,
            skip_validation=False,
            **system_prompt_args,
        )
        # User prompt template
        self.user_prompt_template = prompt_templates.user_prompt_template
        # Examples
        self.examples = []
        if prompt_templates.assistant_prompt_template and prompt_templates.examples:
            for example in prompt_templates.examples:
                self.examples.extend(
                    [
                        {
                            "role": "user",
                            "content": jinja_format(
                                template=self.user_prompt_template,
                                skip_validation=False,
                                **example["input"],
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": jinja_format(
                                template=prompt_templates.assistant_prompt_template,
                                skip_validation=False,
                                **example["output"],
                            ),
                        },
                    ]
                )

    def __call__(self, user_prompt_args: dict[str, str]) -> str | None:
        # Messages
        messages = [{"role": "system", "content": self.system_prompt}]
        # Examples
        messages.extend(self.examples)
        # User prompt
        user_prompt = jinja_format(
            template=self.user_prompt_template,
            skip_validation=False,
            **user_prompt_args,
        )
        messages.append({"role": "user", "content": user_prompt})
        response, _ = self.engine(messages, additional_trace_tags=["judge_llm"])
        return response


class LLMChecker(LLMFunctionBase):
    """
    SoftChecker is a class that extends LLMFunctionBase to utilize a language model to evaluate prompts
    and determine success or failure. It supports multiple voting rounds to increase the reliability of the evaluation.
    The __call__ method returns a boolean indicating the majority vote on the success of the prompt evaluation.
    """

    def __init__(
        self,
        engine: Callable,
        prompt_templates: "LLMFunctionTemplates",
        num_votes: int = 1,
        success_str: str = "[[Success]]",
        failure_str: str = "[[Failure]]",
    ):
        super().__init__(engine, prompt_templates)
        self.judge = LLMFunction(
            engine=engine,
            prompt_templates=prompt_templates,
        )
        # Num votes
        self.num_votes = num_votes
        # Response parsing
        self.success_str = success_str
        self.failure_str = failure_str

    def __call__(self, user_prompt_args: dict[str, str]) -> bool | None:
        # Voting
        votes = []
        for _ in range(self.num_votes):
            # Call the llm
            response = self.judge(user_prompt_args)
            # Parse the response
            if response is None:
                continue
            # Match the verdict markers case-insensitively: both LLM engines in
            # this repo lowercase "True"/"False" in every response they return
            # (LiteLLMEngine.chat_completion, HuggingFaceEngine.chat_completion),
            # so a judge that correctly emits "[[False]]" is seen here as
            # "[[false]]". A case-sensitive match abstains on every vote, and the
            # resulting None is indistinguishable from a genuine rejection at the
            # call site. This mirrors gaia2-cli/core/gaia2_core/judge/checkers.py.
            response_lower = response.lower()
            if self.success_str.lower() in response_lower:
                votes.append(True)
            elif self.failure_str.lower() in response_lower:
                votes.append(False)
            else:
                logger.warning(
                    "LLM checker found neither %s nor %s in the judge response; "
                    "abstaining from this vote. Response: %s",
                    self.success_str,
                    self.failure_str,
                    response[:300],
                )
        if len(votes) == 0:
            return None
        return sum(votes) >= len(votes) / 2


def build_llm_checkers(
    engine: Callable,
    content_checker_prompt_templates: LLMFunctionTemplates = CONTENT_CHECKER_PROMPT_TEMPLATES,
    signature_checker_prompt_templates: LLMFunctionTemplates = SIGNATURE_CHECKER_TEMPLATES,
    sanity_checker_prompt_templates: LLMFunctionTemplates = SANITY_CHECKER_PROMPT_TEMPLATES,
    cab_checker_prompt_templates: LLMFunctionTemplates = CAB_CHECKER_PROMPT_TEMPLATES,
    email_checker_prompt_templates: LLMFunctionTemplates = EMAIL_CHECKER_PROMPT_TEMPLATES,
    message_checker_prompt_templates: LLMFunctionTemplates = MESSAGE_CHECKER_PROMPT_TEMPLATES,
    user_message_checker_prompt_templates: LLMFunctionTemplates = USER_MESSAGE_CHECKER_PROMPT_TEMPLATES,
    event_checker_prompt_templates: LLMFunctionTemplates = EVENT_CHECKER_PROMPT_TEMPLATES,
    tone_checker_prompt_templates: LLMFunctionTemplates = TONE_CHECKER_PROMPT_TEMPLATES,
    num_votes: int = 1,
) -> dict[str, LLMChecker]:
    signature_checker = LLMChecker(
        engine=engine,
        prompt_templates=signature_checker_prompt_templates,
        num_votes=1,
        success_str="[[True]]",
        failure_str="[[False]]",
    )
    # Sanity soft checker
    sanity_checker = LLMChecker(
        engine=engine,
        prompt_templates=sanity_checker_prompt_templates,
        num_votes=1,
        success_str="[[True]]",
        failure_str="[[False]]",
    )
    # Content soft checker
    content_checker = LLMChecker(
        engine=engine,
        prompt_templates=content_checker_prompt_templates,
        num_votes=num_votes,
        success_str="[[Success]]",
        failure_str="[[Failure]]",
    )
    # Cab soft checker
    cab_checker = LLMChecker(
        engine=engine,
        prompt_templates=cab_checker_prompt_templates,
        num_votes=1,
        success_str="[[True]]",
        failure_str="[[False]]",
    )
    # Email soft checker
    email_checker = LLMChecker(
        engine=engine,
        prompt_templates=email_checker_prompt_templates,
        num_votes=num_votes,
        success_str="[[Success]]",
        failure_str="[[Failure]]",
    )
    # Message soft checker
    message_checker = LLMChecker(
        engine=engine,
        prompt_templates=message_checker_prompt_templates,
        num_votes=num_votes,
        success_str="[[Success]]",
        failure_str="[[Failure]]",
    )
    # User message soft checker
    user_message_checker = LLMChecker(
        engine=engine,
        prompt_templates=user_message_checker_prompt_templates,
        num_votes=num_votes,
        success_str="[[Success]]",
        failure_str="[[Failure]]",
    )
    # Event soft checker
    event_checker = LLMChecker(
        engine=engine,
        prompt_templates=event_checker_prompt_templates,
        num_votes=num_votes,
        success_str="[[Success]]",
        failure_str="[[Failure]]",
    )
    # Tone soft checker
    tone_checker = LLMChecker(
        engine=engine,
        prompt_templates=tone_checker_prompt_templates,
        num_votes=num_votes,
        success_str="[[True]]",
        failure_str="[[False]]",
    )

    return {
        SoftCheckerType.signature_checker.value: signature_checker,
        SoftCheckerType.sanity_checker.value: sanity_checker,
        SoftCheckerType.cab_checker.value: cab_checker,
        SoftCheckerType.content_checker.value: content_checker,
        SoftCheckerType.email_checker.value: email_checker,
        SoftCheckerType.message_checker.value: message_checker,
        SoftCheckerType.user_message_checker.value: user_message_checker,
        SoftCheckerType.event_checker.value: event_checker,
        SoftCheckerType.tone_checker.value: tone_checker,
    }


def build_subtask_extractor(
    engine: Callable,
    tool_name: str,
) -> LLMFunction:
    subtask_extractor_prompt_templates = SUBTASK_EXTRACTOR_PROMPT_TEMPLATES
    if "add_calendar_event" in tool_name:
        subtask_extractor_prompt_templates = EVENT_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES
    elif tool_name == "AgentUserInterface__send_message_to_user":
        subtask_extractor_prompt_templates = (
            USER_MESSAGE_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES
        )
    subtask_extractor = LLMFunction(
        engine=engine,
        prompt_templates=subtask_extractor_prompt_templates,
    )
    return subtask_extractor
