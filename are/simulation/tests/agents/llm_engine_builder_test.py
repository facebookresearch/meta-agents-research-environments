# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest
from huggingface_hub import InferenceClient

from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.llm.hf.hf_engine import HuggingFaceLLMEngine
from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder

ENDPOINT = "https://models-proxy.example.com/v1"


@pytest.fixture(autouse=True)
def _inference_token(monkeypatch):
    monkeypatch.setenv("HF_INFERENCE_TOKEN", "not-a-real-token")
    monkeypatch.delenv("HF_BILL_TO", raising=False)


def test_hf_provider_engine_sends_requests_to_the_configured_endpoint(monkeypatch):
    """A configured endpoint must be where the request actually goes.

    `--provider openai` is served by `HuggingFaceLLMEngine`, whose client would
    otherwise route to the provider's own hosted URL and ignore `--endpoint`.
    No request is sent: `_inner_post` records the resolved URL and stops there.
    """
    engine = LLMEngineBuilder()._create_concrete_engine(
        LLMEngineConfig(model_name="gpt-4o-mini", provider="openai", endpoint=ENDPOINT)
    )
    assert isinstance(engine, HuggingFaceLLMEngine)

    recorded = {}

    class RequestRecorded(Exception):
        pass

    def record_request(self, request_parameters, stream=False):
        recorded["url"] = request_parameters.url
        recorded["payload_model"] = request_parameters.json.get("model")
        raise RequestRecorded

    monkeypatch.setattr(InferenceClient, "_inner_post", record_request)

    with pytest.raises(RequestRecorded):
        engine.chat_completion([{"role": "user", "content": "hi"}])

    assert recorded["url"] == f"{ENDPOINT}/chat/completions"
    # The model name still travels in the payload, as an OpenAI-compatible
    # endpoint expects.
    assert recorded["payload_model"] == "gpt-4o-mini"


def test_hf_provider_engine_without_an_endpoint_keeps_the_provider_default():
    """Without an endpoint the client must keep routing to the provider itself."""
    engine = LLMEngineBuilder()._create_concrete_engine(
        LLMEngineConfig(model_name="gpt-4o-mini", provider="openai")
    )
    assert isinstance(engine, HuggingFaceLLMEngine)
    assert engine.model_config.endpoint is None
    # `InferenceClient` stores `base_url` in `model`, and rejects both at once.
    assert engine.client.model is None
