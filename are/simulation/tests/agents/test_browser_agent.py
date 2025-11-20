# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from browser_use import Browser

from are.simulation.agents.custom_agents.browser_agent import BrowserAgent
from are.simulation.agents.llm.mock_llm_engine import MockLLMEngine


class TestBrowserAgent:
    """Test BrowserAgent with external browser injection."""

    def test_browser_agent_creates_own_browser_when_none_provided(self):
        """Test that BrowserAgent creates its own browser when none is provided."""
        llm_engine = MockLLMEngine(model_name="test-model")

        agent = BrowserAgent(llm_engine=llm_engine)

        assert agent.browser is None  # Browser is not created until _start_browser is called
        assert agent._owns_browser is True

    def test_browser_agent_accepts_external_browser(self):
        """Test that BrowserAgent accepts and uses an externally provided browser."""
        llm_engine = MockLLMEngine(model_name="test-model")
        external_browser = Browser()

        agent = BrowserAgent(llm_engine=llm_engine, browser=external_browser)

        assert agent.browser is external_browser
        assert agent._owns_browser is False

    def test_browser_agent_conditional_steps_with_own_browser(self):
        """Test that browser start step is added when agent owns browser."""
        llm_engine = MockLLMEngine(model_name="test-model")

        agent = BrowserAgent(llm_engine=llm_engine)

        # Check that conditional_pre_steps includes browser start step
        step_names = [step.name for step in agent.conditional_pre_steps]
        assert "start_browser_lazy" in step_names
        assert "capture_browser_state" in step_names

    def test_browser_agent_conditional_steps_with_external_browser(self):
        """Test that browser start step is NOT added when using external browser."""
        llm_engine = MockLLMEngine(model_name="test-model")
        external_browser = Browser()

        agent = BrowserAgent(llm_engine=llm_engine, browser=external_browser)

        # Check that conditional_pre_steps does NOT include browser start step
        step_names = [step.name for step in agent.conditional_pre_steps]
        assert "start_browser_lazy" not in step_names
        assert "capture_browser_state" in step_names  # State capture should always be present

    def test_browser_can_be_injected_after_initialization(self):
        """Test that browser can be set after agent initialization."""
        llm_engine = MockLLMEngine(model_name="test-model")
        agent = BrowserAgent(llm_engine=llm_engine)

        # Initially no browser (or None)
        assert agent._owns_browser is True

        # Inject browser later (simulating scenario injection)
        external_browser = Browser()
        agent.browser = external_browser

        assert agent.browser is external_browser
