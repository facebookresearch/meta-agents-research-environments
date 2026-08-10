# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""OmniGAIA judge prompt-template override registry.

Wraps the OmniGAIA-specific (multilingual) checker prompts as
``LLMFunctionTemplates`` so they can be handed straight to
``build_llm_checkers(engine, prompt_template_overrides=...)``.

The dict keys are ``SoftCheckerType`` values; any checker not listed keeps its
gaia2-core default. Select a version by name via
``OMNIGAIA_JUDGE_PROMPT_TEMPLATE_OVERRIDES_REGISTRY``; ``None`` means "use the
gaia2-core defaults unchanged".
"""

from __future__ import annotations

from gaia2_core.judge.config import SoftCheckerType
from gaia2_core.judge.omnigaia_prompts import (
    EMAIL_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
    EVENT_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
    MESSAGE_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
    SIGNATURE_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
    TONE_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
    USER_MESSAGE_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
)
from gaia2_core.judge.prompts import (
    EMAIL_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    EMAIL_CHECKER_EXAMPLES,
    EMAIL_CHECKER_USER_PROMPT_TEMPLATE,
    EVENT_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    EVENT_CHECKER_EXAMPLES,
    EVENT_CHECKER_USER_PROMPT_TEMPLATE,
    MESSAGE_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    MESSAGE_CHECKER_USER_PROMPT_TEMPLATE,
    SIGNATURE_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    SIGNATURE_CHECKER_EXAMPLES,
    SIGNATURE_CHECKER_USER_PROMPT_TEMPLATE,
    TONE_CHECKER_USER_PROMPT_TEMPLATE,
    USER_MESSAGE_CHECKER_USER_PROMPT_TEMPLATE,
    LLMFunctionTemplates,
)

# ── omnigaia: gaia2-core prompts with the English-centric biases removed ────

OMNIGAIA_JUDGE_PROMPT_TEMPLATE_OVERRIDES = {
    SoftCheckerType.user_message_checker.value: LLMFunctionTemplates(
        system_prompt_template=USER_MESSAGE_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
        user_prompt_template=USER_MESSAGE_CHECKER_USER_PROMPT_TEMPLATE,
        assistant_prompt_template=None,
        examples=None,
    ),
    SoftCheckerType.tone_checker.value: LLMFunctionTemplates(
        system_prompt_template=TONE_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
        user_prompt_template=TONE_CHECKER_USER_PROMPT_TEMPLATE,
    ),
    SoftCheckerType.email_checker.value: LLMFunctionTemplates(
        system_prompt_template=EMAIL_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
        user_prompt_template=EMAIL_CHECKER_USER_PROMPT_TEMPLATE,
        assistant_prompt_template=EMAIL_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
        examples=EMAIL_CHECKER_EXAMPLES,
    ),
    SoftCheckerType.message_checker.value: LLMFunctionTemplates(
        system_prompt_template=MESSAGE_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
        user_prompt_template=MESSAGE_CHECKER_USER_PROMPT_TEMPLATE,
        assistant_prompt_template=MESSAGE_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    ),
    SoftCheckerType.event_checker.value: LLMFunctionTemplates(
        system_prompt_template=EVENT_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
        user_prompt_template=EVENT_CHECKER_USER_PROMPT_TEMPLATE,
        assistant_prompt_template=EVENT_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
        examples=EVENT_CHECKER_EXAMPLES,
    ),
    SoftCheckerType.signature_checker.value: LLMFunctionTemplates(
        system_prompt_template=SIGNATURE_CHECKER_SYSTEM_PROMPT_TEMPLATE_OMNIGAIA,
        user_prompt_template=SIGNATURE_CHECKER_USER_PROMPT_TEMPLATE,
        assistant_prompt_template=SIGNATURE_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
        examples=SIGNATURE_CHECKER_EXAMPLES,
    ),
}

# ── Registry ──────────────────────────────────────────────────────────────────

OMNIGAIA_JUDGE_PROMPT_TEMPLATE_OVERRIDES_REGISTRY: dict[str, dict | None] = {
    "default": None,
    "omnigaia": OMNIGAIA_JUDGE_PROMPT_TEMPLATE_OVERRIDES,
}


def resolve_prompt_overrides(version: str | None) -> dict | None:
    """Resolve a judge prompt-version name to its override dict.

    ``None``/empty or ``"default"`` returns ``None`` (gaia2-core defaults).
    Raises ``KeyError`` (with the valid names) for an unknown version.
    """
    if not version or version == "default":
        return None
    try:
        return OMNIGAIA_JUDGE_PROMPT_TEMPLATE_OVERRIDES_REGISTRY[version]
    except KeyError:
        valid = ", ".join(sorted(OMNIGAIA_JUDGE_PROMPT_TEMPLATE_OVERRIDES_REGISTRY))
        raise KeyError(
            f"Unknown judge prompt version {version!r}. Valid: {valid}"
        ) from None
