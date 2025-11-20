# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from are.simulation.apps.gmail import GmailApp
from are.simulation.scenarios.scenario_gmail_browser import GmailBrowserScenario


class TestGmailBrowserScenario:
    """Integration tests for Gmail browser scenario."""

    def test_scenario_creates_browser(self):
        """Test that scenario creates a browser instance."""
        scenario = GmailBrowserScenario()
        scenario.initialize()

        assert scenario.browser is not None
        assert hasattr(scenario, "_loop")

    def test_scenario_shares_browser_with_gmail_app(self):
        """Test that scenario shares browser with Gmail app."""
        scenario = GmailBrowserScenario()
        scenario.initialize()

        # Check browser was created
        assert scenario.browser is not None

        # Check Gmail app has same browser
        gmail_app = next(
            (app for app in scenario.apps if isinstance(app, GmailApp)), None
        )
        assert gmail_app is not None
        assert gmail_app.browser is scenario.browser

    def test_gmail_app_does_not_own_browser(self):
        """Test that Gmail app does not own the browser (scenario owns it)."""
        scenario = GmailBrowserScenario()
        scenario.initialize()

        gmail_app = next(
            (app for app in scenario.apps if isinstance(app, GmailApp)), None
        )
        assert gmail_app is not None
        assert gmail_app._owns_browser is False

    def test_scenario_validation(self):
        """Test that scenario validation works correctly."""
        scenario = GmailBrowserScenario()
        scenario.initialize()

        # Mock environment (not needed for this test scenario)
        env = None

        result = scenario.validate(env)
        assert result.passed is True
        assert "successfully" in result.feedback.lower()

    def test_scenario_get_user_prompt(self):
        """Test that scenario provides a user prompt."""
        scenario = GmailBrowserScenario()

        prompt = scenario.get_user_prompt()
        assert prompt is not None
        assert len(prompt) > 0
        assert "gmail" in prompt.lower()

    def test_scenario_events_flow(self):
        """Test that scenario builds events flow correctly."""
        scenario = GmailBrowserScenario()
        scenario.initialize()
        scenario.build_events_flow()

        assert scenario.events is not None
        assert len(scenario.events) > 0
