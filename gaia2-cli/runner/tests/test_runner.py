# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for ContainerRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from gaia2_runner.launcher import ContainerLauncher
from gaia2_runner.runner import ContainerRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario(
    tmpdir: Path,
    *,
    scenario_id: str = "test_scenario",
    duration: float | None = None,
    events: list | None = None,
    faketime: str | None = None,
) -> Path:
    """Create a minimal scenario JSON and return its path."""
    data: dict = {
        "metadata": {
            "definition": {
                "scenario_id": scenario_id,
            },
        },
    }
    if duration is not None:
        data["metadata"]["definition"]["duration"] = duration
    if faketime is not None:
        data["metadata"]["definition"]["start_time"] = 1728975600.0
    if events is not None:
        data["events"] = events
    path = tmpdir / f"{scenario_id}.json"
    path.write_text(json.dumps(data))
    return path


def _make_user_event(content: str = "Do the task") -> dict:
    """Create a USER event with send_message_to_agent."""
    return {
        "event_type": "USER",
        "event_id": "Event-USER-abc",
        "action": {
            "function": "send_message_to_agent",
            "args": {"content": content},
        },
    }


def _mock_launcher() -> MagicMock:
    """Create a mock ContainerLauncher with sensible defaults."""
    launcher = MagicMock(spec=ContainerLauncher)
    launcher.launch.return_value = "container-123"
    launcher.get_host_adapter_port.return_value = None
    launcher.exec.return_value = ""
    return launcher


def _make_runner(launcher: MagicMock | None = None) -> ContainerRunner:
    """Create a ContainerRunner with a mock launcher."""
    return ContainerRunner(
        launcher=launcher or _mock_launcher(),
        image="test-image:latest",
    )


@pytest.fixture(autouse=True)
def required_judge_env(monkeypatch) -> None:
    """Provide required judge settings for direct runner tests by default."""
    monkeypatch.setenv("GAIA2_JUDGE_MODEL", "judge-model-test")
    monkeypatch.setenv("GAIA2_JUDGE_PROVIDER", "judge-provider-test")


# ---------------------------------------------------------------------------
# TestExtractMessageText
# ---------------------------------------------------------------------------


class TestExtractMessageText:
    def test_plain_string(self) -> None:
        assert ContainerRunner._extract_message_text("hello") == "hello"

    def test_dict_with_string_content(self) -> None:
        assert ContainerRunner._extract_message_text({"content": "hi"}) == "hi"

    def test_dict_with_content_blocks(self) -> None:
        raw = {
            "content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ]
        }
        assert ContainerRunner._extract_message_text(raw) == "line1\nline2"

    def test_dict_with_mixed_blocks(self) -> None:
        raw = {"content": ["plain", {"type": "text", "text": "block"}]}
        assert ContainerRunner._extract_message_text(raw) == "plain\nblock"

    def test_dict_with_error_message(self) -> None:
        raw = {"errorMessage": "something failed"}
        assert ContainerRunner._extract_message_text(raw) == "something failed"

    def test_none_returns_empty(self) -> None:
        assert ContainerRunner._extract_message_text(None) == ""

    def test_number_returns_str(self) -> None:
        assert ContainerRunner._extract_message_text(42) == "42"


# ---------------------------------------------------------------------------
# TestExtractResponseFromEvents
# ---------------------------------------------------------------------------


class TestExtractResponseFromEvents:
    def test_extracts_send_message_to_user(self) -> None:
        events = "\n".join(
            [
                json.dumps(
                    {
                        "app": "EmailClientV2",
                        "fn": "send_email",
                        "args": {},
                        "w": True,
                    }
                ),
                json.dumps(
                    {
                        "app": "AgentUserInterface",
                        "fn": "send_message_to_user",
                        "args": {"content": "Done!"},
                        "w": True,
                    }
                ),
            ]
        )
        assert ContainerRunner._extract_response_from_events(events) == "Done!"

    def test_returns_last_message(self) -> None:
        events = "\n".join(
            [
                json.dumps(
                    {
                        "app": "AgentUserInterface",
                        "fn": "send_message_to_user",
                        "args": {"content": "First"},
                        "w": True,
                    }
                ),
                json.dumps(
                    {
                        "app": "AgentUserInterface",
                        "fn": "send_message_to_user",
                        "args": {"content": "Second"},
                        "w": True,
                    }
                ),
            ]
        )
        assert ContainerRunner._extract_response_from_events(events) == "Second"

    def test_no_message_returns_none(self) -> None:
        events = json.dumps(
            {"app": "Calendar", "fn": "add_event", "args": {}, "w": True}
        )
        assert ContainerRunner._extract_response_from_events(events) is None

    def test_empty_string(self) -> None:
        assert ContainerRunner._extract_response_from_events("") is None

    def test_malformed_json_skipped(self) -> None:
        events = "not json\n" + json.dumps(
            {
                "app": "AgentUserInterface",
                "fn": "send_message_to_user",
                "args": {"content": "ok"},
                "w": True,
            }
        )
        assert ContainerRunner._extract_response_from_events(events) == "ok"


# ---------------------------------------------------------------------------
# TestExtractUserTask
# ---------------------------------------------------------------------------


class TestExtractUserTask:
    def test_current_format(self) -> None:
        runner = _make_runner()
        data = {
            "events": [_make_user_event("Hello agent")],
        }
        assert runner._extract_user_task(data) == "Hello agent"

    def test_legacy_format(self) -> None:
        runner = _make_runner()
        data = {
            "events": [
                {
                    "event_type": "USER",
                    "action": {
                        "function_name": "send_message_to_agent",
                        "args": {"content": "Legacy task"},
                    },
                }
            ],
        }
        assert runner._extract_user_task(data) == "Legacy task"

    def test_list_args_format(self) -> None:
        runner = _make_runner()
        data = {
            "events": [
                {
                    "event_type": "USER",
                    "action": {
                        "function": "send_message_to_agent",
                        "args": [{"name": "content", "value": "List format"}],
                    },
                }
            ],
        }
        assert runner._extract_user_task(data) == "List format"

    def test_no_events(self) -> None:
        runner = _make_runner()
        assert runner._extract_user_task({}) is None

    def test_no_user_event(self) -> None:
        runner = _make_runner()
        data = {
            "events": [
                {
                    "event_type": "AGENT",
                    "action": {"function": "send_email", "args": {}},
                }
            ],
        }
        assert runner._extract_user_task(data) is None


# ---------------------------------------------------------------------------
# TestLoadScenario
# ---------------------------------------------------------------------------


class TestLoadScenario:
    def test_loads_valid_scenario(self, tmp_path: Path) -> None:
        path = _make_scenario(tmp_path, scenario_id="scen_abc")
        runner = _make_runner()
        result_path, data, scenario_id = runner._load_scenario(str(path))
        assert scenario_id == "scen_abc"
        assert data["metadata"]["definition"]["scenario_id"] == "scen_abc"
        assert result_path == path.resolve()

    def test_missing_file_raises(self) -> None:
        runner = _make_runner()
        with pytest.raises(FileNotFoundError):
            runner._load_scenario("/nonexistent/path.json")

    def test_missing_scenario_id_defaults_to_unknown(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.json"
        path.write_text(json.dumps({"metadata": {"definition": {}}}))
        runner = _make_runner()
        _, _, scenario_id = runner._load_scenario(str(path))
        assert scenario_id == "unknown"


# ---------------------------------------------------------------------------
# TestBuildContainerEnv
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestScenarioTimeout
# ---------------------------------------------------------------------------


class TestScenarioTimeout:
    """Test per-scenario timeout from duration field."""

    def _run_with_timeout(
        self,
        tmp_path: Path,
        *,
        duration: float | None = None,
        simulate_timeout: bool = False,
        daemon_judgment: dict | None = None,
        response_timeout: int = 1200,
    ) -> dict:
        """Helper: run a scenario with mocked internals, return result."""
        scenario_path = _make_scenario(
            tmp_path,
            duration=duration,
            events=[_make_user_event()],
        )
        launcher = _mock_launcher()
        runner = _make_runner(launcher)

        # Build daemon status based on test params
        if daemon_judgment is not None:
            daemon_data = {
                "status": "complete",
                "turn": 1,
                "nb_turns": 1,
                "judgment": daemon_judgment,
            }
        elif simulate_timeout:
            daemon_data = {
                "status": "error",
                "turn": 0,
                "nb_turns": 1,
                "num_events": 5,
            }
        else:
            daemon_data = {
                "status": "complete",
                "turn": 1,
                "nb_turns": 1,
                "judgment": {"success": True},
            }

        # Mock _poll_for_response to return immediately
        if simulate_timeout:
            # Simulate elapsed time exceeding the timeout
            def fake_poll(url, timeout=600, **kwargs):
                runner._last_daemon_status = daemon_data
                # Sleep just enough so timed_out = True
                # (poll_elapsed >= effective_timeout - 1)
                return None, "timeout"

            # We need to make poll_elapsed >= effective_timeout - 1
            # Patch time.monotonic to simulate elapsed time
            effective_timeout = (
                max(int(duration) + 60, response_timeout)
                if duration is not None
                else response_timeout
            )
            start_time = 1000.0
            with (
                patch.object(runner, "_poll_for_response", side_effect=fake_poll),
                patch.object(runner, "_collect_events", return_value=("", None)),
                patch.object(runner, "_extract_trace_file"),
                patch.object(runner, "_save_artifacts"),
                patch.object(runner, "_extract_daemon_logs"),
                patch("gaia2_runner.runner.time") as mock_time,
            ):
                mock_time.monotonic.side_effect = [
                    start_time,  # poll_start
                    start_time + effective_timeout,  # poll_elapsed check
                ]
                return runner.run_scenario(
                    str(scenario_path), response_timeout=response_timeout
                )
        else:

            def fake_poll(url, timeout=600, **kwargs):
                runner._last_daemon_status = daemon_data
                return "Agent response", "complete"

            with (
                patch.object(runner, "_poll_for_response", side_effect=fake_poll),
                patch.object(runner, "_collect_events", return_value=("", None)),
                patch.object(runner, "_extract_trace_file"),
                patch.object(runner, "_save_artifacts"),
                patch.object(runner, "_extract_daemon_logs"),
            ):
                return runner.run_scenario(
                    str(scenario_path), response_timeout=response_timeout
                )

    def test_duration_timeout_is_fail(self, tmp_path: Path) -> None:
        """Timeout with scenario duration should be FAIL, not ERROR."""
        result = self._run_with_timeout(tmp_path, duration=330.0, simulate_timeout=True)
        assert result["success"] is False
        assert "Scenario timeout" in result["failure_reasons"][0]
        assert "330s" in result["failure_reasons"][0]

    def test_timeout_without_duration_is_error(self, tmp_path: Path) -> None:
        """Timeout without duration should be ERROR (success=None)."""
        result = self._run_with_timeout(
            tmp_path, duration=None, simulate_timeout=True, response_timeout=600
        )
        assert result["success"] is None
        assert "error" in result

    def test_judgment_pass_not_affected_by_duration(self, tmp_path: Path) -> None:
        """If daemon judged PASS, duration doesn't change the result."""
        result = self._run_with_timeout(
            tmp_path,
            duration=330.0,
            daemon_judgment={"success": True},
        )
        assert result["success"] is True

    def test_judgment_fail_not_affected_by_duration(self, tmp_path: Path) -> None:
        """If daemon judged FAIL, duration doesn't change the result."""
        result = self._run_with_timeout(
            tmp_path,
            duration=330.0,
            daemon_judgment={"success": False, "failure_reason": "wrong args"},
        )
        assert result["success"] is False
        assert "wrong args" in result["failure_reasons"][0]

    def test_no_duration_uses_response_timeout(self, tmp_path: Path) -> None:
        """Without duration, the CLI --timeout value is used."""
        scenario_path = _make_scenario(tmp_path, events=[_make_user_event()])
        launcher = _mock_launcher()
        runner = _make_runner(launcher)

        captured_timeout = {}

        def capture_poll(url, timeout=600, **kwargs):
            captured_timeout["value"] = timeout
            runner._last_daemon_status = {
                "status": "complete",
                "turn": 1,
                "nb_turns": 1,
                "judgment": {"success": True},
            }
            return "ok", "complete"

        with (
            patch.object(runner, "_poll_for_response", side_effect=capture_poll),
            patch.object(runner, "_collect_events", return_value=("", None)),
            patch.object(runner, "_extract_trace_file"),
            patch.object(runner, "_save_artifacts"),
            patch.object(runner, "_extract_daemon_logs"),
        ):
            runner.run_scenario(str(scenario_path), response_timeout=800)

        assert captured_timeout["value"] == 800

    def test_duration_does_not_lower_timeout(self, tmp_path: Path) -> None:
        """With duration=330 and response_timeout=1200, effective timeout should be 1200 (max wins)."""
        scenario_path = _make_scenario(
            tmp_path, duration=330.0, events=[_make_user_event()]
        )
        launcher = _mock_launcher()
        runner = _make_runner(launcher)

        captured_timeout = {}

        def capture_poll(url, timeout=600, **kwargs):
            captured_timeout["value"] = timeout
            runner._last_daemon_status = {
                "status": "complete",
                "turn": 1,
                "nb_turns": 1,
                "judgment": {"success": True},
            }
            return "ok", "complete"

        with (
            patch.object(runner, "_poll_for_response", side_effect=capture_poll),
            patch.object(runner, "_collect_events", return_value=("", None)),
            patch.object(runner, "_extract_trace_file"),
            patch.object(runner, "_save_artifacts"),
            patch.object(runner, "_extract_daemon_logs"),
        ):
            runner.run_scenario(str(scenario_path), response_timeout=1200)

        # max(330+60, 1200) = 1200
        assert captured_timeout["value"] == 1200

    def test_duration_raises_timeout(self, tmp_path: Path) -> None:
        """With duration=330 and response_timeout=200, effective timeout should be 390 (duration+60 wins)."""
        scenario_path = _make_scenario(
            tmp_path, duration=330.0, events=[_make_user_event()]
        )
        launcher = _mock_launcher()
        runner = _make_runner(launcher)

        captured_timeout = {}

        def capture_poll(url, timeout=600, **kwargs):
            captured_timeout["value"] = timeout
            runner._last_daemon_status = {
                "status": "complete",
                "turn": 1,
                "nb_turns": 1,
                "judgment": {"success": True},
            }
            return "ok", "complete"

        with (
            patch.object(runner, "_poll_for_response", side_effect=capture_poll),
            patch.object(runner, "_collect_events", return_value=("", None)),
            patch.object(runner, "_extract_trace_file"),
            patch.object(runner, "_save_artifacts"),
            patch.object(runner, "_extract_daemon_logs"),
        ):
            runner.run_scenario(str(scenario_path), response_timeout=200)

        # max(330+60, 200) = 390
        assert captured_timeout["value"] == 390


class TestBuildContainerEnv:
    def test_uses_required_judge_config_without_judge_type(self) -> None:
        runner = _make_runner()

        env = runner._build_container_env(
            scenario_data={"metadata": {"definition": {}}},
            container_env={"BASE_URL": "https://example.invalid/v1"},
            output_dir=None,
        )

        assert env["GAIA2_JUDGE_FINAL_TURN"] == "1"
        assert env["GAIA2_JUDGE_MODEL"] == "judge-model-test"
        assert env["GAIA2_JUDGE_PROVIDER"] == "judge-provider-test"
        assert "GAIA2_JUDGE_TYPE" not in env

    def test_requires_judge_model_and_provider(self, monkeypatch) -> None:
        monkeypatch.delenv("GAIA2_JUDGE_MODEL", raising=False)
        monkeypatch.delenv("GAIA2_JUDGE_PROVIDER", raising=False)

        runner = _make_runner()

        with pytest.raises(RuntimeError, match="GAIA2_JUDGE_MODEL"):
            runner._build_container_env(
                scenario_data={"metadata": {"definition": {}}},
                container_env={},
                output_dir=None,
            )

    def test_resolves_judge_api_key_from_provider_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "judge-test-key")

        runner = _make_runner()

        env = runner._build_container_env(
            scenario_data={"metadata": {"definition": {}}},
            container_env={
                "GAIA2_JUDGE_MODEL": "judge-model-test",
                "GAIA2_JUDGE_PROVIDER": "openai",
            },
            output_dir=None,
        )

        assert env["GAIA2_JUDGE_API_KEY"] == "judge-test-key"

    def test_does_not_export_provider_specific_judge_keys(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "judge-test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-judge-key")

        runner = _make_runner()

        env = runner._build_container_env(
            scenario_data={"metadata": {"definition": {}}},
            container_env={
                "GAIA2_JUDGE_MODEL": "judge-model-test",
                "GAIA2_JUDGE_PROVIDER": "openai",
            },
            output_dir=None,
        )

        assert env["GAIA2_JUDGE_API_KEY"] == "judge-test-key"
        assert "OPENAI_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env

    def test_forwards_judge_base_url(self) -> None:
        runner = _make_runner()

        env = runner._build_container_env(
            scenario_data={"metadata": {"definition": {}}},
            container_env={
                "GAIA2_JUDGE_MODEL": "judge-model-test",
                "GAIA2_JUDGE_PROVIDER": "openai",
                "GAIA2_JUDGE_BASE_URL": "https://judge.example/v1",
            },
            output_dir=None,
        )

        assert env["GAIA2_JUDGE_BASE_URL"] == "https://judge.example/v1"

    def test_forwards_explicit_judge_api_key(self) -> None:
        runner = _make_runner()

        env = runner._build_container_env(
            scenario_data={"metadata": {"definition": {}}},
            container_env={
                "GAIA2_JUDGE_MODEL": "judge-model-test",
                "GAIA2_JUDGE_PROVIDER": "openai",
                "GAIA2_JUDGE_API_KEY": "judge-key",
            },
            output_dir=None,
        )

        assert env["GAIA2_JUDGE_API_KEY"] == "judge-key"

    def test_resolves_judge_api_key_from_provider_specific_env(
        self, monkeypatch, caplog
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "openai-judge-key")

        runner = _make_runner()

        with caplog.at_level("WARNING"):
            env = runner._build_container_env(
                scenario_data={"metadata": {"definition": {}}},
                container_env={
                    "GAIA2_JUDGE_MODEL": "judge-model-test",
                    "GAIA2_JUDGE_PROVIDER": "openai",
                },
                output_dir=None,
            )

        assert env["GAIA2_JUDGE_API_KEY"] == "openai-judge-key"
        assert "pulling from OPENAI_API_KEY" in caplog.text
