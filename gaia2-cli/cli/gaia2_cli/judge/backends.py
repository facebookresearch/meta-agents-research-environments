# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Select a semantic checker backend independently of the judge algorithm."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from gaia2_core.judge.backends import SoftCheckerFactory

from gaia2_cli.judge.engine import create_litellm_engine


def create_checker_factory(
    model: str,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    extra_body: dict[str, Any] | None = None,
    *,
    audit_path: Path | None = None,
) -> SoftCheckerFactory:
    """Select TypeSafe for ``typesafe``; retain LiteLLM for existing providers.

    New transports implement SoftCheckerFactory without changing the core judge.
    Provider-specific imports are lazy and TypeSafe does not require LiteLLM.
    """
    if (provider or "").lower() == "typesafe":
        from gaia2_cli.judge.typesafe import TypeSafeCheckerFactory

        options = extra_body or {}
        unknown = options.keys() - {"threshold", "timeout"}
        if unknown:
            raise ValueError(
                f"Unknown TypeSafe judge options: {', '.join(sorted(unknown))}"
            )
        return TypeSafeCheckerFactory(
            model=model,
            api_key=api_key,
            base_url=base_url,
            threshold=options.get("threshold", 0.5),
            timeout=options.get("timeout", 30.0),
            audit_path=audit_path,
        )

    engine = create_litellm_engine(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        extra_body=extra_body,
        validate=False,
    )
    if engine is None:
        raise RuntimeError("Judge engine construction returned no engine")
    return _llm_factory(engine)


def _llm_factory(engine: Callable) -> SoftCheckerFactory:
    from gaia2_core.judge.checkers import LLMChecker

    return lambda templates, votes, success, failure: LLMChecker(
        engine, templates, votes, success, failure
    )
