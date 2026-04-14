# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Hard and soft checker helpers for the turn-based judge.

Ported from ``gaia2.validation.tool_judge`` and ``gaia2.validation.utils``.
Optional deps: ``jinja2`` (prompt templates). It gracefully degrades when
unavailable.
"""

from __future__ import annotations

import logging
import os
import re
import string
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable

from gaia2_core.judge.config import CheckerType, SoftCheckerType
from gaia2_core.types import CompletedEvent, OracleEvent, UserDetails

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_str(s: str) -> str:
    """Normalise a string: lowercase, remove accents, strip punctuation.

    Matches ``gaia2.validation.utils.misc.normalize_str``.
    """
    s = s.lower()
    # Remove accents
    s = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s.strip()


def normalize_arg(arg: Any) -> str:
    """Normalise an argument for equality comparison.

    Matches ``gaia2.validation.utils.misc.normalize_arg``.
    """
    return _normalize_str(str(arg))


# ---------------------------------------------------------------------------
# Tag / text extraction
# ---------------------------------------------------------------------------


def extract_text_between_tags(text: str, tag: str) -> list[str]:
    """Extract text between ``<tag>...</tag>`` using regex (replaces bs4 dep)."""
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
    return pattern.findall(text)


# ---------------------------------------------------------------------------
# Jinja rendering
# ---------------------------------------------------------------------------


def jinja_format(template: str, **kwargs: Any) -> str:
    """Render a Jinja2 template.  Falls back to naive replace if jinja2 absent."""
    try:
        from jinja2 import Template

        return Template(template, keep_trailing_newline=True).render(**kwargs)
    except ImportError:
        # Rough fallback for simple templates
        result = template
        for k, v in kwargs.items():
            result = result.replace("{{" + k + "}}", str(v))
        return result


# ---------------------------------------------------------------------------
# Hard checkers — pure Python, no external deps
# ---------------------------------------------------------------------------


def eq_checker(x_agent: Any, x_oracle: Any, **kwargs: Any) -> bool:
    """Exact equality."""
    return x_agent == x_oracle


def unordered_list_checker(
    x_agent: list[Any] | None, x_oracle: list[Any] | None, **kwargs: Any
) -> bool:
    """Set equality; treats ``None`` as empty."""
    if x_agent is None:
        return x_oracle is None or len(x_oracle) == 0
    if x_oracle is None:
        return x_agent is None or len(x_agent) == 0
    return set(x_agent) == set(x_oracle)


def path_checker(x_agent: str | None, x_oracle: str | None, **kwargs: Any) -> bool:
    """Compare after ``os.path.normpath`` + leading-``/`` strip."""
    if not x_agent and not x_oracle:
        return True
    if not x_agent or not x_oracle:
        return False
    normalized_agent = os.path.normpath(x_agent).lstrip("/")
    normalized_oracle = os.path.normpath(x_oracle).lstrip("/")
    return normalized_agent == normalized_oracle


def unordered_path_list_checker(
    x_agent: list[str] | None, x_oracle: list[str] | None, **kwargs: Any
) -> bool:
    """Set of normalised paths."""
    if x_agent is None:
        return x_oracle is None or len(x_oracle) == 0
    if x_oracle is None:
        return x_agent is None or len(x_agent) == 0
    normalized_agent_paths = {os.path.normpath(path).lstrip("/") for path in x_agent}
    normalized_oracle_paths = {os.path.normpath(path).lstrip("/") for path in x_oracle}
    return normalized_agent_paths == normalized_oracle_paths


def _unordered_str_list_with_tolerance_checker(
    x_agent: list[str] | None,
    x_oracle: list[str] | None,
    tolerance_list_str: list[str] | None = None,
    **kwargs: Any,
) -> bool:
    """Normalise strings, remove tolerance items, compare sets."""
    if tolerance_list_str is None:
        tolerance_list_str = []
    _x_agent = [_normalize_str(x) for x in x_agent] if x_agent is not None else []
    _x_oracle = [_normalize_str(x) for x in x_oracle] if x_oracle is not None else []
    _x_agent = [x for x in _x_agent if x not in tolerance_list_str]
    _x_oracle = [x for x in _x_oracle if x not in tolerance_list_str]
    return set(_x_agent) == set(_x_oracle)


def list_attendees_checker(
    x_agent: list[str] | None,
    x_oracle: list[str] | None,
    tolerance_list_str: list[str] | None = None,
    **kwargs: Any,
) -> bool:
    """Tolerance-aware unordered string-list comparison."""
    if tolerance_list_str is None:
        tolerance_list_str = []
    _tolerance_list = [_normalize_str(x) for x in tolerance_list_str]
    # If the oracle list is empty or contains only tolerance strings, pass
    if x_oracle is None or len(x_oracle) == 0:
        return True
    if all(
        _normalize_str(x_oracle[i]) in _tolerance_list for i in range(len(x_oracle))
    ):
        return True
    return _unordered_str_list_with_tolerance_checker(
        x_agent, x_oracle, _tolerance_list
    )


def datetime_checker(x_agent: str | None, x_oracle: str | None, **kwargs: Any) -> bool:
    """Parse ``%Y-%m-%d %H:%M:%S`` and compare."""
    if x_agent is None or x_oracle is None:
        return x_agent == x_oracle
    _x_agent = datetime.strptime(x_agent, "%Y-%m-%d %H:%M:%S")
    _x_oracle = datetime.strptime(x_oracle, "%Y-%m-%d %H:%M:%S")
    return _x_agent == _x_oracle


def eq_str_strip_checker(
    x_agent: str | None, x_oracle: str | None, **kwargs: Any
) -> bool:
    """Strip + equality."""
    _x_agent = x_agent.strip() if bool(x_agent) else ""
    _x_oracle = x_oracle.strip() if bool(x_oracle) else ""
    return _x_agent == _x_oracle


def phone_number_checker(
    x_agent: str | None, x_oracle: str | None, **kwargs: Any
) -> bool:
    """Digits-only comparison; treats ``None`` and ``""`` as equivalent."""
    _x_agent = "".join(char for char in x_agent if char.isdigit()) if x_agent else ""
    _x_oracle = "".join(char for char in x_oracle if char.isdigit()) if x_oracle else ""
    return _x_agent == _x_oracle


def contain_any_checker(x_agent: str, targets: list[str], **kwargs: Any) -> bool:
    """Case-insensitive substring match (any target)."""
    return any(x_oracle.lower() in x_agent.lower() for x_oracle in targets)


def contain_all_checker(x_agent: str, targets: list[str], **kwargs: Any) -> bool:
    """Case-insensitive substring match (all targets)."""
    return all(x_oracle.lower() in x_agent.lower() for x_oracle in targets)


# ---------------------------------------------------------------------------
# Hard-checker registry
# ---------------------------------------------------------------------------

_HARD_CHECKERS: dict[str, Callable] = {
    CheckerType.eq_checker.value: eq_checker,
    CheckerType.unordered_list_checker.value: unordered_list_checker,
    CheckerType.datetime_checker.value: datetime_checker,
    CheckerType.list_attendees_checker.value: list_attendees_checker,
    CheckerType.phone_number_checker.value: phone_number_checker,
    CheckerType.eq_str_strip_checker.value: eq_str_strip_checker,
    CheckerType.path_checker.value: path_checker,
    CheckerType.unordered_path_list_checker.value: unordered_path_list_checker,
    CheckerType.contain_any_checker.value: contain_any_checker,
    CheckerType.contain_all_checker.value: contain_all_checker,
}


# ---------------------------------------------------------------------------
# hard_compare
# ---------------------------------------------------------------------------


def hard_compare(
    agent_args: dict[str, Any],
    oracle_args: dict[str, Any],
    arg_to_checker_type: dict[str, CheckerType],
    tolerance_list_str: list[str] | None = None,
) -> tuple[bool, str]:
    """Run hard checkers on all registered args.

    Returns:
        ``(passed, rationale)`` — rationale is empty string on success.
    """
    for arg_name, check_type in arg_to_checker_type.items():
        if not check_type.is_hard() and not check_type.is_scripted():
            continue  # skip llm_checker entries
        checker_fn = _HARD_CHECKERS.get(check_type.value)
        if checker_fn is None:
            continue
        agent_val = agent_args.get(arg_name)
        oracle_val = oracle_args.get(arg_name)

        # Build extra kwargs the checker might need
        extra_kwargs: dict[str, Any] = {"arg_name": arg_name}
        if check_type == CheckerType.list_attendees_checker:
            extra_kwargs["tolerance_list_str"] = tolerance_list_str

        # Scripted checkers (contain_any, contain_all) compare agent vs targets
        if check_type.is_scripted():
            if not checker_fn(agent_val, oracle_val, **extra_kwargs):
                return (
                    False,
                    f"{check_type.value}:{arg_name} (agent={agent_val})",
                )
        else:
            if not checker_fn(x_agent=agent_val, x_oracle=oracle_val, **extra_kwargs):
                return (
                    False,
                    f"{check_type.value}:{arg_name} "
                    f"(agent={agent_val}, oracle={oracle_val})",
                )
    return (True, "")


# ---------------------------------------------------------------------------
# LLMFunction / LLMChecker (from llm_utils.py)
# ---------------------------------------------------------------------------


class LLMFunction:
    """Format prompts and call LLM engine.  Returns raw string."""

    def __init__(self, engine: Callable, prompt_templates: Any) -> None:
        system_prompt_args = prompt_templates.system_prompt_args or {}
        self.system_prompt = jinja_format(
            prompt_templates.system_prompt_template, **system_prompt_args
        )
        self.user_prompt_template = prompt_templates.user_prompt_template
        self.examples: list[dict[str, str]] = []
        if prompt_templates.assistant_prompt_template and prompt_templates.examples:
            for example in prompt_templates.examples:
                self.examples.extend(
                    [
                        {
                            "role": "user",
                            "content": jinja_format(
                                self.user_prompt_template, **example["input"]
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": jinja_format(
                                prompt_templates.assistant_prompt_template,
                                **example["output"],
                            ),
                        },
                    ]
                )
        self.engine = engine

    def __call__(self, user_prompt_args: dict[str, str]) -> str | None:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(self.examples)
        user_prompt = jinja_format(self.user_prompt_template, **user_prompt_args)
        messages.append({"role": "user", "content": user_prompt})
        response, _ = self.engine(messages)
        return response


class LLMChecker:
    """Majority-vote LLM checker.  Returns ``bool`` or ``None``."""

    def __init__(
        self,
        engine: Callable,
        prompt_templates: Any,
        num_votes: int = 1,
        success_str: str = "[[Success]]",
        failure_str: str = "[[Failure]]",
    ) -> None:
        self.judge = LLMFunction(engine=engine, prompt_templates=prompt_templates)
        self.num_votes = num_votes
        self.success_str = success_str
        self.failure_str = failure_str
        self.last_response: str | None = None

    def __call__(self, user_prompt_args: dict[str, str]) -> bool | None:
        votes: list[bool] = []
        for _ in range(self.num_votes):
            response = self.judge(user_prompt_args)
            self.last_response = response
            if response is None:
                logger.debug("LLM checker returned None (engine failure)")
                continue
            response_lower = response.lower()
            if self.success_str.lower() in response_lower:
                votes.append(True)
            elif self.failure_str.lower() in response_lower:
                votes.append(False)
                logger.info(
                    "LLM checker voted FAIL (looking for %s/%s): %s",
                    self.success_str,
                    self.failure_str,
                    response[:300],
                )
            else:
                logger.info(
                    "LLM checker vote ABSTAIN (no %s or %s found): %s",
                    self.success_str,
                    self.failure_str,
                    response[:300],
                )
        if not votes:
            return None
        result = sum(votes) >= len(votes) / 2
        logger.debug("LLM checker result: %s (votes=%s)", result, votes)
        return result


# ---------------------------------------------------------------------------
# build_llm_checkers
# ---------------------------------------------------------------------------


def build_llm_checkers(engine: Callable, num_votes: int = 1) -> dict[str, LLMChecker]:
    """Build all LLM checker instances."""
    from gaia2_core.judge import prompts as P

    return {
        SoftCheckerType.signature_checker.value: LLMChecker(
            engine, P.SIGNATURE_CHECKER_TEMPLATES, 1, "[[True]]", "[[False]]"
        ),
        SoftCheckerType.sanity_checker.value: LLMChecker(
            engine,
            P.SANITY_CHECKER_PROMPT_TEMPLATES,
            1,
            "[[True]]",
            "[[False]]",
        ),
        SoftCheckerType.content_checker.value: LLMChecker(
            engine, P.CONTENT_CHECKER_PROMPT_TEMPLATES, num_votes
        ),
        SoftCheckerType.cab_checker.value: LLMChecker(
            engine,
            P.CAB_CHECKER_PROMPT_TEMPLATES,
            1,
            "[[True]]",
            "[[False]]",
        ),
        SoftCheckerType.email_checker.value: LLMChecker(
            engine, P.EMAIL_CHECKER_PROMPT_TEMPLATES, num_votes
        ),
        SoftCheckerType.message_checker.value: LLMChecker(
            engine, P.MESSAGE_CHECKER_PROMPT_TEMPLATES, num_votes
        ),
        SoftCheckerType.user_message_checker.value: LLMChecker(
            engine, P.USER_MESSAGE_CHECKER_PROMPT_TEMPLATES, num_votes
        ),
        SoftCheckerType.event_checker.value: LLMChecker(
            engine, P.EVENT_CHECKER_PROMPT_TEMPLATES, num_votes
        ),
        SoftCheckerType.tone_checker.value: LLMChecker(
            engine,
            P.TONE_CHECKER_PROMPT_TEMPLATES,
            num_votes,
            "[[True]]",
            "[[False]]",
        ),
    }


# ---------------------------------------------------------------------------
# build_subtask_extractor
# ---------------------------------------------------------------------------


def build_subtask_extractor(engine: Callable, tool_name: str) -> LLMFunction:
    """Build the subtask extractor for a given tool."""
    from gaia2_core.judge import prompts as P

    templates = P.SUBTASK_EXTRACTOR_PROMPT_TEMPLATES
    if "add_calendar_event" in tool_name:
        templates = P.EVENT_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES
    elif tool_name == "AgentUserInterface__send_message_to_user":
        templates = P.USER_MESSAGE_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES
    return LLMFunction(engine=engine, prompt_templates=templates)


# ---------------------------------------------------------------------------
# placeholder_checker (pure Python, no LLM)
# ---------------------------------------------------------------------------


def placeholder_checker(agent_args: dict[str, Any], **kwargs: Any) -> bool:
    """Check for leftover placeholders in agent args."""
    agent_args_str = " ".join(str(v) for v in agent_args.values())
    placeholders = [
        "[User's Name]",
        "[User Name]",
        "[User]",
        "[Your Name]",
        "[My Name]",
        "Best regards,\nYour Name",
        "Best,\nYour Name",
    ]
    return not any(p.lower() in agent_args_str.lower() for p in placeholders)


# ---------------------------------------------------------------------------
# Helpers for soft_compare
# ---------------------------------------------------------------------------


def _describe_action_args(args: dict[str, Any]) -> str:
    """Format args for LLM prompt display."""
    return "\n".join(f"{k}: {v}" for k, v in args.items()).strip()


def _equality_checker(
    agent_args: dict[str, Any],
    oracle_args: dict[str, Any],
) -> bool:
    """Normalised equality across all oracle arg keys."""
    for arg_name in oracle_args:
        if arg_name not in agent_args:
            return False
        if normalize_arg(agent_args[arg_name]) != normalize_arg(oracle_args[arg_name]):
            return False
    return True


# ---------------------------------------------------------------------------
# soft_compare
# ---------------------------------------------------------------------------


def soft_compare(
    agent_args: dict[str, Any],
    oracle_args: dict[str, Any],
    tool_name: str,
    soft_checker_types: list[SoftCheckerType],
    llm_checkers: dict[str, LLMChecker] | None,
    tasks: list[str] | None = None,
    user_details: UserDetails | None = None,
    oracle_event_time: float | None = None,
    scenario_start_time: float | None = None,
    engine: Callable | None = None,
) -> tuple[bool | None, str, str]:
    """Run soft (LLM) checkers.

    Returns:
        ``(passed_or_none, rationale, judge_output)``
    """
    if not soft_checker_types:
        return (True, "", "")

    # Filter to LLM-checker args only (those with CheckerType.llm_checker)
    # by this point we already know what oracle_args/agent_args are

    # Try equality first — skip LLM if args already match
    if _equality_checker(agent_args=agent_args, oracle_args=oracle_args):
        return (True, "", "")

    # No LLM engine available → hard-only fallback
    if llm_checkers is None:
        return (True, "no LLM, hard-only", "")

    # Compute common kwargs for soft checkers
    today_date = ""
    effective_time = oracle_event_time
    if effective_time is None or effective_time <= 0:
        effective_time = scenario_start_time
    if effective_time is not None and effective_time > 0:
        today_date = datetime.fromtimestamp(effective_time, tz=timezone.utc).strftime(
            "%Y-%m-%d %A"
        )

    user_name = (
        f"{user_details.first_name} {user_details.last_name}" if user_details else ""
    )
    user_address = f"{user_details.address}" if user_details else ""

    if tasks is None:
        tasks = [""]
    task = tasks[-1] if tasks else ""
    previous_task = "\n".join(tasks[:-1]) if len(tasks) > 1 else ""

    # Subtask extraction (only if any checker needs it)
    need_subtask = any(c.need_subtask for c in soft_checker_types)
    subtask = ""
    if need_subtask and engine is not None:
        subtask_extractor = build_subtask_extractor(engine=engine, tool_name=tool_name)
        full_task = "\n".join(tasks).strip()
        if full_task:
            raw = subtask_extractor(
                user_prompt_args={
                    "tool_name": tool_name,
                    "oracle_action_call": _describe_action_args(oracle_args),
                    "task": full_task,
                }
            )
            if raw is not None:
                extracted = extract_text_between_tags(raw, "subtask")
                subtask = extracted[0].strip() if extracted else ""

    agent_action_call = _describe_action_args(agent_args)
    oracle_action_call = _describe_action_args(oracle_args)

    # Apply each soft checker in order
    for checker_type in soft_checker_types:
        result: bool | None = None

        if checker_type == SoftCheckerType.placeholder_checker:
            result = placeholder_checker(agent_args=agent_args)

        elif checker_type == SoftCheckerType.signature_checker:
            result = llm_checkers[SoftCheckerType.signature_checker.value](
                user_prompt_args={
                    "agent_action_call": agent_action_call,
                    "user_name": user_name,
                }
            )

        elif checker_type == SoftCheckerType.sanity_checker:
            # Shortcut for numerical values
            is_numerical = re.match(r"^content: \d+(\.\d+)?$", agent_action_call)
            if is_numerical:
                result = True
            else:
                result = llm_checkers[SoftCheckerType.sanity_checker.value](
                    user_prompt_args={
                        "agent_action_call": agent_action_call,
                        "task": "\n".join([previous_task, task]),
                    }
                )

        elif checker_type == SoftCheckerType.content_checker:
            result = llm_checkers[SoftCheckerType.content_checker.value](
                user_prompt_args={
                    "agent_action_call": agent_action_call,
                    "oracle_action_call": oracle_action_call,
                    "task": subtask,
                    "tool_name": tool_name,
                    "today_date": today_date,
                    "user_address": user_address,
                }
            )

        elif checker_type == SoftCheckerType.cab_checker:
            # Replace "Home" placeholder in both oracle and agent args
            # with user address before the LLM call — deterministic,
            # no need for LLM to interpret the substitution.
            _home_variants = ("home", "my place", "my address")
            resolved_oracle = dict(oracle_args)
            resolved_agent = dict(agent_args)
            if user_address:
                for args in (resolved_oracle, resolved_agent):
                    for k in ("start_location", "end_location"):
                        if str(args.get(k, "")).strip().lower() in _home_variants:
                            args[k] = user_address
            resolved_oracle_call = _describe_action_args(resolved_oracle)
            resolved_agent_call = _describe_action_args(resolved_agent)
            result = llm_checkers[SoftCheckerType.cab_checker.value](
                user_prompt_args={
                    "agent_action_call": resolved_agent_call,
                    "oracle_action_call": resolved_oracle_call,
                    "user_address": user_address,
                }
            )

        elif checker_type == SoftCheckerType.email_checker:
            result = llm_checkers[SoftCheckerType.email_checker.value](
                user_prompt_args={
                    "agent_action_call": agent_action_call,
                    "oracle_action_call": oracle_action_call,
                    "today_date": today_date,
                }
            )

        elif checker_type == SoftCheckerType.message_checker:
            result = llm_checkers[SoftCheckerType.message_checker.value](
                user_prompt_args={
                    "agent_action_call": agent_action_call,
                    "oracle_action_call": oracle_action_call,
                    "today_date": today_date,
                }
            )

        elif checker_type == SoftCheckerType.user_message_checker:
            result = llm_checkers[SoftCheckerType.user_message_checker.value](
                user_prompt_args={
                    "agent_action_call": agent_action_call,
                    "oracle_action_call": oracle_action_call,
                    "task": subtask,
                }
            )

        elif checker_type == SoftCheckerType.event_checker:
            result = llm_checkers[SoftCheckerType.event_checker.value](
                user_prompt_args={
                    "agent_action_call": agent_action_call,
                    "oracle_action_call": oracle_action_call,
                    "user_address": user_address,
                    "task": subtask,
                }
            )

        elif checker_type == SoftCheckerType.tone_checker:
            result = llm_checkers[SoftCheckerType.tone_checker.value](
                user_prompt_args={
                    "agent_action_call": agent_action_call,
                }
            )

        # Evaluate result
        judge_output = ""
        if llm_checkers and checker_type.value in llm_checkers:
            judge_output = llm_checkers[checker_type.value].last_response or ""
        if result is None:
            return (None, f"{checker_type.value}:inconclusive", judge_output)
        if not result:
            logger.info(
                "Soft checker '%s' failed for tool '%s'. Agent args: %s",
                checker_type.value,
                tool_name,
                agent_args,
            )
            return (
                False,
                f"{checker_type.value} (agent_args={agent_args})",
                judge_output,
            )

    return (True, "", "")


# ---------------------------------------------------------------------------
# mild_compare — combined hard + soft
# ---------------------------------------------------------------------------


def mild_compare(
    agent_event: CompletedEvent,
    oracle_event: OracleEvent,
    arg_checker_registry: dict[str, dict[str, CheckerType]],
    soft_checker_registry: dict[str, list[SoftCheckerType]],
    llm_checkers: dict[str, LLMChecker] | None = None,
    tolerance_list_str: list[str] | None = None,
    tasks: list[str] | None = None,
    user_details: UserDetails | None = None,
    engine: Callable | None = None,
    scenario_start_time: float | None = None,
) -> tuple[bool | None, str, str]:
    """Combined hard + soft comparison.

    Returns:
        ``(passed_or_none, rationale, judge_output)``
    """
    # --- tool name match ---
    if oracle_event.tool_name != agent_event.tool_name:
        return (
            False,
            f"tool name mismatch: agent={agent_event.tool_name}, "
            f"oracle={oracle_event.tool_name}",
            "",
        )

    tool_name = oracle_event.tool_name
    agent_args = agent_event.get_args()
    oracle_args = oracle_event.get_args()

    # --- hard compare ---
    arg_to_checker_type = arg_checker_registry.get(tool_name, {})
    hard_passed, hard_rationale = hard_compare(
        agent_args=agent_args,
        oracle_args=oracle_args,
        arg_to_checker_type=arg_to_checker_type,
        tolerance_list_str=tolerance_list_str,
    )
    if not hard_passed:
        return (False, hard_rationale, "")

    # --- soft compare ---
    soft_checker_types = soft_checker_registry.get(tool_name, [])
    if not soft_checker_types:
        return (True, "", "")

    # Filter to llm_checker args only for soft comparison
    selected_action_args = [
        name
        for name, check_type in arg_to_checker_type.items()
        if check_type == CheckerType.llm_checker
    ]
    if not selected_action_args:
        return (True, "", "")

    # Filter args to only those with content
    selected_action_args = [
        arg for arg in selected_action_args if bool(oracle_args.get(arg))
    ]
    if not selected_action_args:
        return (True, "", "")

    filtered_oracle = {
        k: v for k, v in oracle_args.items() if k in selected_action_args
    }
    filtered_agent = {k: v for k, v in agent_args.items() if k in selected_action_args}

    return soft_compare(
        agent_args=filtered_agent,
        oracle_args=filtered_oracle,
        tool_name=tool_name,
        soft_checker_types=soft_checker_types,
        llm_checkers=llm_checkers,
        tasks=tasks,
        user_details=user_details,
        oracle_event_time=oracle_event.event_time,
        scenario_start_time=scenario_start_time,
        engine=engine,
    )
