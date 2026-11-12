# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for exception-run accounting across the two result writers.

A scenario run that dies with an exception (for example an auth error raised by
the LLM engine) must be reported as an exception run by both writers:

* ``output.jsonl``, built from :func:`get_scenario_result_info`
* ``benchmark_stats.json``, built from :func:`calculate_statistics`

Previously ``ScenarioRunner.run`` rewrote ``success=None`` to ``success=False``
after the run had already been classified, so crashed runs were reported with
``status="failed"`` next to a populated ``exception_type`` and were counted in
the success-rate denominator, while ``exception_runs`` stayed at 0.
"""

from unittest.mock import patch

from are.simulation.benchmark.hf_upload_utils import get_scenario_result_info
from are.simulation.benchmark.report_stats import calculate_statistics
from are.simulation.scenario_runner import ScenarioRunner, ScenarioRunnerConfig
from are.simulation.scenarios.config import MultiScenarioRunnerConfig
from are.simulation.scenarios.scenario import Scenario
from are.simulation.scenarios.validation_result import (
    MultiScenarioValidationResult,
    ScenarioValidationResult,
)


class LLMEngineAuthError(Exception):
    """Stand-in for the LLMEngineException("Auth error in litellm.") in the report."""


def _run_crashing_scenario(scenario_id: str) -> ScenarioValidationResult:
    """Run a scenario whose execution raises, through the real ScenarioRunner.run."""
    scenario = Scenario()
    scenario.scenario_id = scenario_id
    runner = ScenarioRunner()
    config = ScenarioRunnerConfig(export=False)
    with patch.object(
        ScenarioRunner,
        "_run",
        side_effect=LLMEngineAuthError("Auth error in litellm."),
    ):
        return runner.run(config, scenario)


def _stats(results: list[tuple[str, ScenarioValidationResult]]) -> dict:
    multi = MultiScenarioValidationResult(
        run_config=MultiScenarioRunnerConfig(model="m", model_provider="p")
    )
    for scenario_id, result in results:
        multi.add_result(result, scenario_id, 1)
    df = multi.to_polars({"config": "search", "phase_name": "standard"})
    return calculate_statistics(df)["global"]


def test_crashed_run_keeps_success_none():
    """A run that raised is not a validated failure."""
    result = _run_crashing_scenario("crashed")

    assert isinstance(result.exception, LLMEngineAuthError)
    assert result.success is None


def test_crashed_run_is_reported_as_exception_in_output_jsonl():
    """output.jsonl must not claim status="failed" for a run that raised."""
    result = _run_crashing_scenario("crashed")

    score, status = get_scenario_result_info(result)

    assert status == "exception"
    assert score is None


def test_crashed_run_is_counted_in_exception_runs():
    """benchmark_stats.json must report the crashed run under exception_runs."""
    stats = _stats([("crashed", _run_crashing_scenario("crashed"))])

    assert stats["exception_runs"] == 1
    assert stats["failed_runs"] == 0
    assert stats["validated_runs"] == 0


def test_crashed_runs_are_excluded_from_success_rate_denominator():
    """Crashed runs must not deflate the published pass rate.

    Mirrors the run in the bug report: 15 runs, 2 passes, 7 genuine failures
    and 6 runs killed by an auth error. The pass rate is 2/9, not 2/15.
    """
    results: list[tuple[str, ScenarioValidationResult]] = []
    for i in range(2):
        results.append((f"pass_{i}", ScenarioValidationResult(success=True)))
    for i in range(7):
        results.append((f"fail_{i}", ScenarioValidationResult(success=False)))
    for i in range(6):
        results.append((f"crash_{i}", _run_crashing_scenario(f"crash_{i}")))

    stats = _stats(results)

    assert stats["total_runs"] == 15
    assert stats["exception_runs"] == 6
    assert stats["validated_runs"] == 9
    assert stats["failed_runs"] == 7
    assert stats["success_runs"] == 2
    assert stats["micro_success_rate"] == 2 / 9 * 100

    # Contract asserted by tests/cli/test_report_stats.py::test_run_type_assumptions
    assert stats["total_runs"] == (
        stats["validated_runs"] + stats["exception_runs"] + stats["no_validation_runs"]
    )


def test_writers_agree_on_the_number_of_exception_runs():
    """The two writers must not disagree on how many runs raised."""
    results = [(f"crash_{i}", _run_crashing_scenario(f"crash_{i}")) for i in range(3)]
    results.append(("pass_0", ScenarioValidationResult(success=True)))

    from_output_jsonl = sum(
        1 for _, result in results if get_scenario_result_info(result)[1] == "exception"
    )

    assert from_output_jsonl == _stats(results)["exception_runs"] == 3
