# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
from typing import Any

from huggingface_hub import (
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    InferenceClient,
)
from pydantic import BaseModel
from requests.exceptions import HTTPError

from are.simulation.agents.llm.llm_engine import LLMEngine
from are.simulation.agents.llm.types import MessageRole
from are.simulation.agents.multimodal import Attachment
from are.simulation.core.reliability_utils import retryable

logger = logging.getLogger(__name__)

# Retryable config
BACKOFF_SECONDS = int(os.getenv("LLMENGINE_BACKOFF_SECONDS", "3"))
ATTEMPTS = int(os.getenv("LLMENGINE_ATTEMPTS", "5"))


def _is_retryable_http_error(error: Exception) -> bool:
    """Check if an HTTP error should be retried based on status code."""
    # Only apply this logic to HTTPError instances
    if not isinstance(error, HTTPError):
        return False

    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        status_code = error.response.status_code
        # Don't retry 4xx client errors (400-499)
        # These indicate client-side issues that won't be resolved by retrying
        return not (400 <= status_code < 500)
    # If we can't determine the status code, err on the side of retrying
    return True


class HuggingFaceModelConfig(BaseModel):
    model_name: str
    provider: str


class HuggingFaceLLMEngine(LLMEngine):
    """
    A class that extends the LLMEngine to provide a specific implementation for
    models hosted on Hugging Face's platform. It utilizes the Hugging Face
    InferenceClient.
    Attributes:
        model_config (HuggingFaceModelConfig): The configuration for the model.
    """

    def __init__(self, model_config: HuggingFaceModelConfig):
        super().__init__(model_config.model_name)

        self.model_config = model_config

        # api_key will need to be parametrizable to the provider's token to be more general
        api_key = os.getenv("HF_INFERENCE_TOKEN") or os.getenv("HF_TOKEN")

        self.client = InferenceClient(
            bill_to=os.getenv("HF_BILL_TO") or None,
            provider=model_config.provider,  # type: ignore
            api_key=api_key,
        )

    def _convert_message_to_hf_format(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert a message to HuggingFace format, handling both text and multimodal content."""
        role = MessageRole(message["role"]).value
        if role == "tool-response":
            role = "user"

        # Handle attachments if present
        attachments: list[Attachment] | None = message.get("attachments")
        content = message.get("content", "")

        if attachments and len(attachments) > 0:
            # Create multimodal content with both text and images
            content_list = []

            # Add text content if present
            if content:
                content_list.append({"type": "text", "text": content})

            # Add image attachments
            for attachment in attachments:
                if attachment.mime.startswith("image/"):
                    content_list.append(attachment.to_openai_json())
                else:
                    logger.warning(
                        f"Unsupported attachment mime type: {attachment.mime}"
                    )

            return {"role": role, "content": content_list}
        else:
            # Text-only message
            return {"role": role, "content": content}

    @retryable(
        n_attempts=ATTEMPTS,
        sleep_time_s=BACKOFF_SECONDS,
        exceptions=(HTTPError,),
        retry_condition=_is_retryable_http_error,
    )
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stop_sequences=[],
        **kwargs,
    ) -> tuple[str, dict | None]:
        # Convert messages to HuggingFace format with multimodal support
        converted_messages = []
        for message in messages:
            converted_message = self._convert_message_to_hf_format(message)
            converted_messages.append(converted_message)

        response = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=converted_messages,
            stop=stop_sequences,
        )

        if not isinstance(response, ChatCompletionOutput):
            error_msg = f"Expected ChatCompletionOutput, got {type(response)}"
            logger.error(f"{error_msg}. Response: {response}")
            raise TypeError(error_msg)

        if not response.choices:
            error_msg = "Expected at least 1 choice in response"
            logger.error(f"{error_msg}. Response: {response}")
            raise ValueError(error_msg)

        if not isinstance(response.choices[0], ChatCompletionOutputComplete):
            error_msg = f"Expected ChatCompletionOutputComplete for first choice, got {type(response.choices[0])}"
            logger.error(f"{error_msg}. Choice: {response.choices[0]}")
            raise TypeError(error_msg)

        content = response.choices[0].message.content
        if content is None:
            error_msg = "Expected content in message, got None"
            logger.error(f"{error_msg}. Message: {response.choices[0].message}")
            raise ValueError(error_msg)

        return content, None
