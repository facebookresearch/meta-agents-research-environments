# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Adapter wiring upstream ``OpenAICompatInferencer`` into the contract's
``LLMClient`` protocol.

The term-extractor Pass A takes one already-formatted (system, prompt) pair
per scenario, whereas ``OpenAICompatInferencer`` is template-based: it formats
both at call-time from a kwargs dict. The adapter wraps the inferencer in a
fixed ``"{prompt}"`` / ``"{system}"`` template so the caller can pass the
already-formatted strings via the kwargs dict.
"""

from __future__ import annotations

from gaia2_mt.llm.client import OpenAICompatInferencer


# Trivial templates — we feed the already-formatted strings through verbatim.
_PASSTHROUGH_SYSTEM = "{system}"
_PASSTHROUGH_PROMPT = "{prompt}"


class OpenAICompatLLMClient:
    """Adapt ``OpenAICompatInferencer`` to the contract ``LLMClient`` protocol.

    The contract's term extractor calls ``infer_one(system, prompt)`` with two
    already-formatted strings per scenario; this wrapper bridges that to the
    upstream template-based inferencer.
    """

    def __init__(
        self,
        model_name: str,
        max_concurrency: int = 8,
        max_retries: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self._inferencer = OpenAICompatInferencer(
            system_prompt=_PASSTHROUGH_SYSTEM,
            prompt_template=_PASSTHROUGH_PROMPT,
            model_name=model_name,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            temperature=temperature,
        )

    def infer_one(self, system: str, prompt: str) -> str | None:
        return self._inferencer.infer({"system": system, "prompt": prompt})

    def infer_batch(self, pairs: list[tuple[str, str]]) -> list[str | None]:
        """Batch version — useful when running Pass A over many scenarios."""
        kwargs_list = [{"system": s, "prompt": p} for s, p in pairs]
        return self._inferencer.infer_batch(kwargs_list)
