# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Transport-independent contracts for replaceable semantic checkers.

A backend judges one rendered rubric at a time. Graph matching, argument
checks, temporal checks and aggregation remain owned by the judge.
"""

from __future__ import annotations

from typing import Protocol

from gaia2_core.judge.prompts import LLMFunctionTemplates


class SoftChecker(Protocol):
    """A semantic verdict plus inspectable output for failed comparisons."""

    last_response: str | None

    def __call__(self, user_prompt_args: dict[str, str]) -> bool | None:
        """Return a verdict, or None when a verdict is unavailable."""
        ...


class SoftCheckerFactory(Protocol):
    """Build a checker for a rubric, independently of its API transport."""

    def __call__(
        self,
        prompt_templates: LLMFunctionTemplates,
        num_votes: int,
        success_str: str,
        failure_str: str,
        /,
    ) -> SoftChecker: ...
