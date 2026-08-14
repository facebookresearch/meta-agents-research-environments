# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Async OpenAI-compatible LLM caller."""

from __future__ import annotations

import asyncio
import logging
import os

from openai import APIStatusError, AsyncOpenAI, BadRequestError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

from gaia2_mt.llm.config import DEFAULT_REVIEW_MODEL, resolve_endpoint


logger: logging.Logger = logging.getLogger(__name__)


def _resolve_api_key() -> str:
    """Resolve the LLM API key from ``GAIA2_MT_LLM_API_KEY``.

    The models are served by a local OpenAI-compatible server, which accepts any
    non-empty value.
    """
    value = os.environ.get("GAIA2_MT_LLM_API_KEY")
    if value and value.strip():
        return value.strip()
    raise RuntimeError(
        "No LLM API key found. Set GAIA2_MT_LLM_API_KEY — vLLM accepts any "
        "non-empty value, e.g. GAIA2_MT_LLM_API_KEY=EMPTY."
    )


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception should trigger a retry."""
    if isinstance(exc, BadRequestError):
        return False
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code < 500:
        return False
    return True


# Reasoning-tuned chat templates that prepend ``<think>`` to every assistant
# turn unless we explicitly disable it. Disabling avoids reasoning traces
# polluting the strict JSON parser downstream.
# Includes both HuggingFace-style repo names (Qwen/Qwen3*) and the shorter ids
# servers often advertise (qwen3*, qwen36*). Without the short-id entries, MT
# against qwen3.6-fp8 sends
# requests WITHOUT chat_template_kwargs.enable_thinking=False, the hybrid template
# emits <think>...</think> that either poisons JSON parsing (retries → None → source
# passthrough) or blows past max_model_len (HTTP 400 "prompt contains at least
# 131073 input tokens"). Discovered 2026-07-07 when qwen MT for spa silently wrote
# unchanged English into spa_Latn/data.
_DISABLE_THINKING_PREFIXES: tuple[str, ...] = (
    "Qwen/Qwen3",
    "qwen/qwen3",
    "qwen3",
    "qwen36",
)


def _extra_body_for_model(model_name: str) -> dict | None:
    """Return per-model ``extra_body`` kwargs for chat.completions, or None.

    Currently injects ``chat_template_kwargs.enable_thinking=False`` for Qwen3
    family models, which otherwise emit a ``<think>…</think>`` block before
    any JSON output.
    """
    if model_name.startswith(_DISABLE_THINKING_PREFIXES):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


class OpenAICompatInferencer:
    """Async OpenAI-compatible LLM caller.

    General-purpose inferencer: takes a system prompt template and a user
    prompt template, formats them with kwargs, and batches requests with
    configurable concurrency and retries.
    """

    def __init__(
        self,
        system_prompt: str,
        prompt_template: str,
        model_name: str = DEFAULT_REVIEW_MODEL,
        max_concurrency: int = 10,
        max_retries: int = 5,
        temperature: float = 0.0,
    ):
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.temperature = temperature

        self._api_key = _resolve_api_key()
        self._base_url = resolve_endpoint(model_name)

    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    async def _call_one(self, client: AsyncOpenAI, kwargs: dict) -> str | None:
        prompt = self.prompt_template.format(**kwargs)
        system = self.system_prompt.format(**kwargs)

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=64),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.INFO),
            reraise=True,
        )
        async def _do_call() -> str:
            extra_body = _extra_body_for_model(self.model_name)
            kwargs = dict(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            resp = await client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        async with self.semaphore:
            try:
                return await _do_call()
            except Exception as e:
                logger.error(f"LLM call failed after retries, returning None: {e}")
                return None

    async def _infer_batch_async(
        self, prompt_kwargs_list: list[dict]
    ) -> list[str | None]:
        client = self._create_client()
        try:

            async def _indexed_call(idx: int, kw: dict) -> tuple[int, str]:
                result = await self._call_one(client, kw)
                return idx, result

            indexed_tasks = [
                _indexed_call(i, kw) for i, kw in enumerate(prompt_kwargs_list)
            ]
            results: list[str | None] = [None] * len(indexed_tasks)
            for coro in tqdm(
                asyncio.as_completed(indexed_tasks),
                total=len(indexed_tasks),
                desc="LLM calls",
            ):
                idx, result = await coro
                results[idx] = result
            return results  # type: ignore[return-value]
        finally:
            await client.close()

    def _run_async(self, coro: asyncio.coroutines) -> any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)

        return asyncio.run(coro)

    def infer(self, prompt_kwargs: dict) -> str:
        """Run a single inference call."""
        return self._run_async(self._infer_one_async(prompt_kwargs))

    async def _infer_one_async(self, prompt_kwargs: dict) -> str:
        client = self._create_client()
        try:
            return await self._call_one(client, prompt_kwargs)
        finally:
            await client.close()

    def infer_batch(self, prompt_kwargs_list: list[dict]) -> list[str | None]:
        """Run a batch of inference calls with concurrency."""
        return self._run_async(self._infer_batch_async(prompt_kwargs_list))
