# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import os
from abc import ABC, abstractmethod

from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.llm.llm_engine import LLMEngine


class AbstractLLMEngineBuilder(ABC):
    """
    Abstract class for building LLM engines.
    """

    @abstractmethod
    def create_engine(
        self,
        engine_config: LLMEngineConfig,
        mock_responses: list[str] | None = None,
    ) -> LLMEngine:
        """
        Method to create an LLM engine.
        :param agent_config: Configuration for the agent to be built.
        :param mock_responses: Optional list of mock responses to be used for the engine.
        :returns: An instance of the LLM engine.
        """


class LLMEngineBuilder(AbstractLLMEngineBuilder):
    def create_engine(
        self,
        engine_config: LLMEngineConfig,
        mock_responses: list[str] | None = None,
    ) -> LLMEngine:
        llm_engine = self._create_concrete_engine(engine_config)

        if mock_responses is not None:
            from are.simulation.agents.llm.llm_engine import MockLLMEngine

            llm_engine = MockLLMEngine(mock_responses, llm_engine)

        return llm_engine

    def _create_concrete_engine(
        self,
        engine_config: LLMEngineConfig,
    ) -> LLMEngine:
        """
        Private method to create an LLM engine, can be overridden by subclasses.
        :param engine_config: Configuration for the agent to be built.
        :returns: An instance of the LLM engine.
        """
        if engine_config.provider in ["llama-api"]:
            return self._create_llama_api_engine(engine_config)

        if engine_config.provider in ["local", "mock"]:
            return self._create_local_engine(engine_config)

        if engine_config.provider == "huggingface":
            return self._create_huggingface_engine(engine_config)

        if engine_config.provider in [
            "black-forest-labs",
            "cerebras",
            "cohere",
            "fal-ai",
            "featherless-ai",
            "fireworks-ai",
            "groq",
            "hf-inference",
            "hyperbolic",
            "nebius",
            "novita",
            "nscale",
            "openai",
            "replicate",
            "sambanova",
            "together",
        ]:
            return self._create_hf_provider_engine(engine_config)

        # Fallback to generic LiteLLM engine for unknown providers
        return self._create_generic_litellm_engine(engine_config)

    def _create_llama_api_engine(self, engine_config: LLMEngineConfig) -> LLMEngine:
        """
        Create an LLM engine for the llama-api provider.
        :param engine_config: Configuration for the engine.
        :returns: An instance of the LLM engine.
        """
        from are.simulation.agents.llm.litellm.litellm_engine import (
            LiteLLMEngine,
            LiteLLMModelConfig,
        )

        endpoint = os.environ.get("LLAMA_API_BASE", "https://api.llama.com/compat/v1")
        key = os.environ.get("LLAMA_API_KEY")
        if key is None:
            raise EnvironmentError("LLAMA_API_KEY must be set in the environment")

        model_config = LiteLLMModelConfig(
            model_name=engine_config.model_name,
            provider="openai",
            endpoint=endpoint,
            api_key=key,
        )
        return LiteLLMEngine(model_config=model_config)

    def _create_local_engine(self, engine_config: LLMEngineConfig) -> LLMEngine:
        """
        Create an LLM engine for local or mock providers.
        :param engine_config: Configuration for the engine.
        :returns: An instance of the LLM engine.
        """
        from are.simulation.agents.llm.litellm.litellm_engine import (
            LiteLLMEngine,
            LiteLLMModelConfig,
        )

        # Ensure provider is not None before passing it to LiteLLMModelConfig
        provider = engine_config.provider or "local"
        model_config = LiteLLMModelConfig(
            model_name=engine_config.model_name,
            provider=provider,
            endpoint=engine_config.endpoint,
        )
        return LiteLLMEngine(model_config=model_config)

    def _create_huggingface_engine(self, engine_config: LLMEngineConfig) -> LLMEngine:
        """
        Create an LLM engine for the huggingface provider.
        :param engine_config: Configuration for the engine.
        :returns: An instance of the LLM engine.
        """
        from are.simulation.agents.llm.litellm.litellm_engine import (
            LiteLLMEngine,
            LiteLLMModelConfig,
        )

        provider = "local"
        model_name = f"huggingface/{engine_config.model_name}"
        model_config = LiteLLMModelConfig(
            model_name=model_name,
            provider=provider,
            endpoint=engine_config.endpoint,
        )
        return LiteLLMEngine(model_config=model_config)

    def _create_hf_provider_engine(self, engine_config: LLMEngineConfig) -> LLMEngine:
        """
        Create an LLM engine for various Hugging Face providers.
        :param engine_config: Configuration for the engine.
        :returns: An instance of the LLM engine.
        """
        # Note: would be better to grab the list dynamically with
        # from huggingface_hub.inference._providers import PROVIDER_T
        from are.simulation.agents.llm.hf.hf_engine import (
            HuggingFaceLLMEngine,
            HuggingFaceModelConfig,
        )

        # Ensure provider is not None before passing it to HuggingFaceModelConfig
        provider = engine_config.provider or "default"
        model_config = HuggingFaceModelConfig(
            model_name=engine_config.model_name,
            provider=provider,
            endpoint=engine_config.endpoint,
        )
        return HuggingFaceLLMEngine(model_config=model_config)

    def _create_generic_litellm_engine(
        self, engine_config: LLMEngineConfig
    ) -> LLMEngine:
        """
        Create a generic LLM engine using LiteLLM for any provider.
        This method does not attempt to acquire API keys, leaving that to LiteLLM.
        :param engine_config: Configuration for the engine.
        :returns: An instance of the LLM engine.
        """
        from are.simulation.agents.llm.litellm.litellm_engine import (
            LiteLLMEngine,
            LiteLLMModelConfig,
        )

        # Ensure provider is not None before passing it to LiteLLMModelConfig
        assert engine_config.provider, "provider must be set in engine_config"
        model_config = LiteLLMModelConfig(
            model_name=engine_config.model_name,
            provider=engine_config.provider,
            endpoint=engine_config.endpoint,
        )
        return LiteLLMEngine(model_config=model_config)
