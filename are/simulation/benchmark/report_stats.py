# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from are.simulation.scenarios.validation_result import MultiScenarioValidationResult

logger = logging.getLogger(__name__)


def _safe_mean_to_float(mean_value: Any) -> float | None:
    """Safely convert polars mean() result to float or None.

    Polars .mean() can return PythonLiteral types including date objects,
    but we expect numeric values for success rates.

    :param mean_value: Result from polars .mean() operation
    :returns: Float value or None if conversion fails
    """
    if mean_value is None:
        return None
    if isinstance(mean_value, (int, float)):
        return float(mean_value)
    # If it's not a numeric type, return None to indicate invalid data
    return None


def combine_results_to_dataframe(
    results: dict[tuple[str, str, float, bool, bool], "MultiScenarioValidationResult"],
) -> pl.DataFrame:
    """Convert multiple MultiScenarioValidationResult objects to a single polars DataFrame.

    This function uses the to_polars() method of each MultiScenarioValidationResult and
    adds metadata columns for phase, config, etc.

    :param results: Dictionary mapping (phase_name, config, a2a_app_prop, has_tool_augmentation, has_env_events) tuples to MultiScenarioValidationResult objects
    :type results: dict[tuple[str, str, float, bool, bool], MultiScenarioValidationResult]
    :returns: Polars DataFrame with columns for each scenario run plus metadata
    :rtype: pl.DataFrame
    """
    import polars as pl

    dataframes = []

    for result_key, multi_result in results.items():
        phase_name, config, a2a_app_prop, has_tool_augmentation, has_env_events = (
            result_key
        )

        # Use the to_polars() method with extra metadata columns
        df = multi_result.to_polars(
            {
                "phase_name": str(phase_name),
                "config": str(config),
                "a2a_app_prop": str(a2a_app_prop),
                "has_tool_augmentation": str(has_tool_augmentation),
                "has_env_events": str(has_env_events),
            }
        )

        if not df.is_empty():
            dataframes.append(df)

    if not dataframes:
        # Return empty DataFrame with expected schema
        return pl.DataFrame(
            schema={
                "base_scenario_id": pl.Utf8,
                "run_number": pl.Int64,
                "success_numeric": pl.Float64,
                "success_bool": pl.Boolean,
                "status": pl.Utf8,
                "has_exception": pl.Boolean,
                "exception_type": pl.Utf8,
                "exception_message": pl.Utf8,
                "rationale": pl.Utf8,
                "export_path": pl.Utf8,
                "model": pl.Utf8,
                "model_provider": pl.Utf8,
                "agent": pl.Utf8,
                "phase_name": pl.Utf8,
                "config": pl.Utf8,
                "a2a_app_prop": pl.Float64,
                "has_tool_augmentation": pl.Boolean,
                "has_env_events": pl.Boolean,
                "run_duration": pl.Float64,
                "job_duration": pl.Float64,
            }
        )

    # Concatenate all DataFrames
    return pl.concat(dataframes, how="vertical")


def generate_validation_report_header(model: str, model_provider: str) -> str:
    """Generate the header section of a validation report.

    :param model: Model name
    :type model: str
    :param model_provider: Model provider
    :type model_provider: str
    :returns: Formatted header string
    :rtype: str
    """
    header = "\n=== GAIA2 Validation Report ===\n"
    header += f"Model: {model}\n"
    header += f"Provider: {model_provider}\n\n"
    return header


def _count_runs_by_type(df: pl.DataFrame) -> dict[str, int]:
    """Count different types of runs in the dataframe.

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :returns: Dictionary with run counts by type
    :rtype: dict[str, int]
    """
    if df.is_empty():
        return {
            "total_runs": 0,
            "validated_runs": 0,
            "success_runs": 0,
            "failed_runs": 0,
            "exception_runs": 0,
            "no_validation_runs": 0,
        }

    total_runs = len(df)
    validated_runs = len(df.filter(pl.col("success_numeric").is_not_null()))
    success_runs = len(df.filter(pl.col("status") == "success"))
    failed_runs = len(df.filter(pl.col("status") == "failed"))
    exception_runs = len(df.filter(pl.col("status") == "exception"))
    no_validation_runs = len(df.filter(pl.col("status") == "no_validation"))

    return {
        "total_runs": total_runs,
        "validated_runs": validated_runs,
        "success_runs": success_runs,
        "failed_runs": failed_runs,
        "exception_runs": exception_runs,
        "no_validation_runs": no_validation_runs,
    }


def _calculate_success_rate_stats(df: pl.DataFrame) -> dict[str, float]:
    """Calculate success rate statistics from validated runs only.

    Success rate is calculated only from runs with non-null success_numeric values.
    Exception runs and no validation runs both have a null success_numeric and
    are therefore excluded from success rate calculations.

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :returns: Dictionary with success rate statistics
    :rtype: dict[str, float]
    """
    # Filter to only validated runs (non-null success_numeric)
    validated_df = df.filter(pl.col("success_numeric").is_not_null())

    if validated_df.is_empty():
        return {
            "success_rate": 0.0,
            "success_rate_std": 0.0,
            "success_rate_sem": 0.0,
        }

    # Calculate overall success rate from individual runs
    success_values = validated_df.select("success_numeric").to_series()
    mean_success = _safe_mean_to_float(success_values.mean())
    success_rate = (mean_success * 100.0) if mean_success is not None else 0.0

    # Calculate STD across run-level means (not individual runs)
    run_numbers = validated_df.select("run_number").unique().to_series().sort()
    run_level_means = []

    for run_num in run_numbers:
        run_df = validated_df.filter(pl.col("run_number") == run_num)
        if not run_df.is_empty():
            run_mean = _safe_mean_to_float(
                run_df.select("success_numeric").to_series().mean()
            )
            if run_mean is not None:
                run_level_means.append(run_mean * 100.0)

    # Standard deviation across run-level means
    success_rate_std = (
        float(np.std(run_level_means, ddof=1)) if len(run_level_means) > 1 else 0.0
    )

    # Standard error of the mean
    success_rate_sem = (
        success_rate_std / float(np.sqrt(len(run_level_means)))
        if len(run_level_means) > 1
        else 0.0
    )

    return {
        "success_rate": success_rate,
        "success_rate_std": success_rate_std,
        "success_rate_sem": success_rate_sem,
    }


def _calculate_pass_at_k_stats(df: pl.DataFrame) -> dict[str, Any]:
    """Calculate Pass@k and Pass^k statistics.

    Pass@k: scenarios with at least 1 success
    Pass^k: scenarios with all successes

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :returns: Dictionary with pass@k statistics
    :rtype: dict[str, Any]
    """
    # Filter to only validated runs for pass@k calculations
    validated_df = df.filter(pl.col("success_numeric").is_not_null())

    if validated_df.is_empty():
        total_scenarios = df.select(["base_scenario_id", "phase_name"]).n_unique()
        return {
            "pass_at_k": 0,
            "pass_at_k_percent": 0.0,
            "pass_k": 0,
            "pass_k_percent": 0.0,
            "total_scenarios": total_scenarios,
        }

    # Count total unique scenarios (from original df, not just validated)
    total_scenarios = df.select(["base_scenario_id", "phase_name"]).n_unique()

    # Calculate per-scenario success rates
    scenario_stats = validated_df.group_by(["base_scenario_id", "phase_name"]).agg(
        [
            pl.col("success_numeric").mean().alias("scenario_success_rate"),
        ]
    )

    # Pass@k: scenarios with at least one success (success_rate > 0)
    pass_at_k = len(scenario_stats.filter(pl.col("scenario_success_rate") > 0.0))

    # Pass^k: scenarios with all successes (success_rate = 1.0)
    pass_k = len(scenario_stats.filter(pl.col("scenario_success_rate") == 1.0))

    return {
        "pass_at_k": pass_at_k,
        "pass_at_k_percent": (
            (pass_at_k / total_scenarios * 100) if total_scenarios > 0 else 0.0
        ),
        "pass_k": pass_k,
        "pass_k_percent": (
            (pass_k / total_scenarios * 100) if total_scenarios > 0 else 0.0
        ),
        "total_scenarios": total_scenarios,
    }


def _calculate_run_duration_stats(df: pl.DataFrame) -> dict[str, float]:
    """Calculate run duration statistics from runs with non-null duration values.

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :returns: Dictionary with run duration statistics
    :rtype: dict[str, float]
    """
    # Filter to only runs with non-null run_duration
    duration_df = df.filter(pl.col("run_duration").is_not_null())

    if duration_df.is_empty():
        return {
            "avg_run_duration": 0.0,
            "avg_run_duration_std": 0.0,
        }

    # Calculate overall average run duration
    avg_duration = _safe_mean_to_float(
        duration_df.select("run_duration").to_series().mean()
    )
    avg_duration = avg_duration if avg_duration is not None else 0.0

    # Calculate standard deviation of run durations
    duration_std = _safe_mean_to_float(
        duration_df.select("run_duration").to_series().std()
    )
    duration_std = duration_std if duration_std is not None else 0.0

    return {
        "avg_run_duration": avg_duration,
        "avg_run_duration_std": duration_std,
    }


def _calculate_capability_stats(df: pl.DataFrame, capability: str) -> dict[str, Any]:
    """Calculate statistics for a single capability.

    :param df: DataFrame with scenario results for this capability
    :type df: pl.DataFrame
    :param capability: Capability name
    :type capability: str
    :returns: Dictionary with capability statistics
    :rtype: dict[str, Any]
    """
    run_counts = _count_runs_by_type(df)
    success_rate_stats = _calculate_success_rate_stats(df)
    pass_k_stats = _calculate_pass_at_k_stats(df)
    duration_stats = _calculate_run_duration_stats(df)

    return {
        "capability": capability,
        **run_counts,
        **success_rate_stats,
        **pass_k_stats,
        **duration_stats,
    }


def _calculate_cross_run_stats(
    df: pl.DataFrame, capability_stats: dict[str, dict[str, Any]], aggregation_type: str
) -> dict[str, float]:
    """Calculate cross-run statistics with different aggregation methods.

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :param capability_stats: Dictionary of capability statistics
    :type capability_stats: dict[str, dict[str, Any]]
    :param aggregation_type: Either "macro" (unweighted average) or "micro" (weighted by runs)
    :type aggregation_type: str
    :returns: Dictionary with cross-run statistics
    :rtype: dict[str, float]
    """
    validated_df = df.filter(pl.col("success_numeric").is_not_null())

    if validated_df.is_empty():
        return {
            f"{aggregation_type}_success_rate": 0.0,
            f"{aggregation_type}_success_rate_std": 0.0,
            f"{aggregation_type}_success_rate_sem": 0.0,
        }

    # Calculate overall success rate using specified aggregation
    capability_success_rates = [
        stats["success_rate"] for stats in capability_stats.values()
    ]

    if aggregation_type == "macro":
        # Macro: Average of capability success rates (each capability weighted equally)
        overall_success_rate = (
            float(np.mean(capability_success_rates))
            if capability_success_rates
            else 0.0
        )
    elif aggregation_type == "micro":
        # Micro: Success rate weighted by number of validated runs per capability
        capability_run_counts = [
            stats["validated_runs"] for stats in capability_stats.values()
        ]
        total_validated_runs = sum(capability_run_counts)
        if total_validated_runs > 0:
            overall_success_rate = (
                sum(
                    rate * count
                    for rate, count in zip(
                        capability_success_rates, capability_run_counts
                    )
                )
                / total_validated_runs
            )
        else:
            overall_success_rate = 0.0
    else:
        raise ValueError(f"Unknown aggregation_type: {aggregation_type}")

    # Calculate STD across run-level scores
    run_numbers = validated_df.select("run_number").unique().to_series().sort()
    run_level_scores = []

    for run_num in run_numbers:
        run_df = validated_df.filter(pl.col("run_number") == run_num)
        run_capability_scores = []
        run_capability_counts = []

        # Calculate success rate and count for each capability in this run
        # Group by both config and phase_name to match the per_capability grouping
        for config_phase in (
            validated_df.select(["config", "phase_name"]).unique().iter_rows()
        ):
            config, phase_name = config_phase
            config_phase_run_df = run_df.filter(
                (pl.col("config") == config) & (pl.col("phase_name") == phase_name)
            )
            if not config_phase_run_df.is_empty():
                run_capability_score = _safe_mean_to_float(
                    config_phase_run_df.select("success_numeric").to_series().mean()
                )
                if run_capability_score is not None:
                    run_capability_scores.append(run_capability_score * 100.0)
                    run_capability_counts.append(len(config_phase_run_df))

        # Aggregate across capabilities for this run
        if run_capability_scores:
            if aggregation_type == "macro":
                # Macro: unweighted average across capabilities
                run_score = float(np.mean(run_capability_scores))
            elif aggregation_type == "micro":
                # Micro: weighted average across capabilities
                if sum(run_capability_counts) > 0:
                    run_score = sum(
                        score * count
                        for score, count in zip(
                            run_capability_scores, run_capability_counts
                        )
                    ) / sum(run_capability_counts)
                else:
                    continue  # Skip this run if no valid counts

            run_level_scores.append(run_score)

    # Calculate STD and SEM across run-level scores
    success_rate_std = (
        float(np.std(run_level_scores, ddof=1)) if len(run_level_scores) > 1 else 0.0
    )

    success_rate_sem = (
        success_rate_std / float(np.sqrt(len(run_level_scores)))
        if len(run_level_scores) > 1
        else 0.0
    )

    return {
        f"{aggregation_type}_success_rate": overall_success_rate,
        f"{aggregation_type}_success_rate_std": success_rate_std,
        f"{aggregation_type}_success_rate_sem": success_rate_sem,
    }


def _calculate_global_macro_stats(
    df: pl.DataFrame, capability_stats: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Calculate global macro statistics (average of capability success rates).

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :param capability_stats: Dictionary of capability statistics
    :type capability_stats: dict[str, dict[str, Any]]
    :returns: Dictionary with macro statistics
    :rtype: dict[str, float]
    """
    return _calculate_cross_run_stats(df, capability_stats, "macro")


def _calculate_global_micro_stats(
    df: pl.DataFrame, capability_stats: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Calculate global micro statistics (weighted by runs per capability).

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :param capability_stats: Dictionary of capability statistics
    :type capability_stats: dict[str, dict[str, Any]]
    :returns: Dictionary with micro statistics
    :rtype: dict[str, float]
    """
    return _calculate_cross_run_stats(df, capability_stats, "micro")


def calculate_statistics(df: pl.DataFrame) -> dict[str, Any]:
    """Calculate comprehensive statistics for the benchmark results.

    This is the main function that calculates all statistics consistently
    for both JSON and text reports.

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :returns: Dictionary with comprehensive statistics
    :rtype: dict[str, Any]
    """
    if df.is_empty():
        return {
            "per_capability": {},
            "global": {
                "total_runs": 0,
                "validated_runs": 0,
                "success_runs": 0,
                "failed_runs": 0,
                "exception_runs": 0,
                "no_validation_runs": 0,
                "total_scenarios": 0,
                "macro_success_rate": 0.0,
                "macro_success_rate_std": 0.0,
                "macro_success_rate_sem": 0.0,
                "micro_success_rate": 0.0,
                "micro_success_rate_std": 0.0,
                "micro_success_rate_sem": 0.0,
                "pass_at_k": 0,
                "pass_at_k_percent": 0.0,
                "pass_k": 0,
                "pass_k_percent": 0.0,
                "job_duration": 0.0,
            },
        }

    # Calculate per-capability statistics, grouped by config and phase_name
    per_capability = {}
    for config_phase in df.select(["config", "phase_name"]).unique().iter_rows():
        config, phase_name = config_phase
        config_phase_df = df.filter(
            (pl.col("config") == config) & (pl.col("phase_name") == phase_name)
        )

        # Create a key that combines config and phase for mini capability
        if config == "mini":
            capability_key = f"{config}_{phase_name}"
            display_name = f"{config} ({phase_name})"
        else:
            capability_key = config
            display_name = config

        per_capability[capability_key] = _calculate_capability_stats(
            config_phase_df, display_name
        )

    # Calculate global statistics
    global_run_counts = _count_runs_by_type(df)
    global_pass_k_stats = _calculate_pass_at_k_stats(df)
    global_macro_stats = _calculate_global_macro_stats(df, per_capability)
    global_micro_stats = _calculate_global_micro_stats(df, per_capability)
    global_duration_stats = _calculate_run_duration_stats(df)

    return {
        "per_capability": per_capability,
        "global": {
            **global_run_counts,
            **global_pass_k_stats,
            **global_macro_stats,
            **global_micro_stats,
            **global_duration_stats,
            "job_duration": df.select("job_duration").to_series().mean(),
        },
    }


def generate_validation_report_content(
    df: pl.DataFrame,
    num_runs: int = 3,
    header_format: str = "===",
    header_prefix: str = "",
    header_suffix: str = "",
) -> str:
    """Generate the content section of a validation report.

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :param num_runs: Number of runs per scenario for display
    :type num_runs: int
    :param header_format: Format for section headers ("===", "###", etc.)
    :type header_format: str
    :param header_prefix: Prefix to add before header text
    :type header_prefix: str
    :param header_suffix: Suffix to add after header text
    :type header_suffix: str
    :returns: Formatted content string
    :rtype: str
    """
    stats = calculate_statistics(df)

    # Per-capability statistics
    capability_display_names = {
        "execution": "Execution",
        "search": "Search",
        "adaptability": "Adaptability",
        "time": "Time",
        "ambiguity": "Ambiguity",
        "mini": "Mini (Mixed Capabilities)",
    }

    content = ""

    for capability, cap_stats in stats["per_capability"].items():
        display_name = capability_display_names.get(capability, capability.title())

        # Format header based on parameters
        if header_format == "===":
            content += f"\n=== {display_name} ===\n"
        elif header_format == "###":
            content += f"\n### {header_prefix}{display_name}{header_suffix}\n\n"
        elif header_format == "####":
            content += f"\n#### {header_prefix}{display_name}{header_suffix}\n\n"
        else:
            content += (
                f"\n{header_format} {header_prefix}{display_name}{header_suffix}\n"
            )

        # Show total runs and breakdown
        content += f"  - Scenarios: {cap_stats['total_scenarios']} unique ({cap_stats['total_runs']} total runs)\n"

        # Show breakdown of run types if there are non-validated runs
        if cap_stats["no_validation_runs"] > 0 or cap_stats["exception_runs"] > 0:
            content += f"    • Validated runs (counted in success rate): {cap_stats['validated_runs']}\n"
            if cap_stats["no_validation_runs"] > 0:
                content += f"    • No validation runs (not counted): {cap_stats['no_validation_runs']}\n"
            if cap_stats["exception_runs"] > 0:
                content += f"    • Exception runs (counted as failures): {cap_stats['exception_runs']}\n"

        content += f"  - Success rate: {cap_stats['success_rate']:.1f}% ± {cap_stats['success_rate_sem']:.1f}% (STD: {cap_stats['success_rate_std']:.1f}%)\n"
        content += f"  - Pass@{num_runs}: {cap_stats['pass_at_k']} scenarios ({cap_stats['pass_at_k_percent']:.1f}%)\n"
        content += f"  - Pass^{num_runs}: {cap_stats['pass_k']} scenarios ({cap_stats['pass_k_percent']:.1f}%)\n"
        content += f"  - Average run duration: {cap_stats['avg_run_duration']:.1f}s (STD: {cap_stats['avg_run_duration_std']:.1f}s)\n"

    # Global summary section
    global_stats = stats["global"]

    if header_format == "===":
        content += "\n=== Global Summary ===\n"
    elif header_format == "###":
        content += f"\n### {header_prefix}Global Summary{header_suffix}\n\n"
    elif header_format == "####":
        content += f"\n#### {header_prefix}Global Summary{header_suffix}\n\n"
    else:
        content += f"\n{header_format} {header_prefix}Global Summary{header_suffix}\n"

    # Show total runs and breakdown
    content += f"  - Scenarios: {global_stats['total_scenarios']} unique ({global_stats['total_runs']} total runs)\n"

    # Show breakdown of run types if there are non-validated runs
    if global_stats["no_validation_runs"] > 0 or global_stats["exception_runs"] > 0:
        content += f"    • Validated runs (counted in success rate): {global_stats['validated_runs']}\n"
        if global_stats["no_validation_runs"] > 0:
            content += f"    • No validation runs (not counted): {global_stats['no_validation_runs']}\n"
        if global_stats["exception_runs"] > 0:
            content += f"    • Exception runs (counted as failures): {global_stats['exception_runs']}\n"

    content += f"  - Macro success rate: {global_stats['macro_success_rate']:.1f}% ± {global_stats['macro_success_rate_sem']:.1f}% (STD: {global_stats['macro_success_rate_std']:.1f}%)\n"
    content += f"  - Micro success rate: {global_stats['micro_success_rate']:.1f}% ± {global_stats['micro_success_rate_sem']:.1f}% (STD: {global_stats['micro_success_rate_std']:.1f}%)\n"
    content += f"  - Pass@{num_runs}: {global_stats['pass_at_k']} scenarios ({global_stats['pass_at_k_percent']:.1f}%)\n"
    content += f"  - Pass^{num_runs}: {global_stats['pass_k']} scenarios ({global_stats['pass_k_percent']:.1f}%)\n"
    content += f"  - Average run duration: {global_stats['avg_run_duration']:.1f}s (STD: {global_stats['avg_run_duration_std']:.1f}s)\n"
    content += f"  - Job duration: {global_stats['job_duration']:.1f} seconds\n"

    return content


def generate_validation_report(
    df: pl.DataFrame, model: str, model_provider: str, num_runs: int = 3
) -> str:
    """Generate a validation report using custom statistics as the new standard.

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :param model: Model name
    :type model: str
    :param model_provider: Model provider
    :type model_provider: str
    :param num_runs: Number of runs per scenario for display
    :type num_runs: int
    :returns: Formatted report string
    :rtype: str
    """
    header = generate_validation_report_header(model, model_provider)
    content = generate_validation_report_content(df, num_runs)
    return header + content


def generate_json_stats_report(
    df: pl.DataFrame, model: str, model_provider: str
) -> dict[str, Any]:
    """Generate a computer-readable JSON report.

    :param df: DataFrame with scenario results
    :type df: pl.DataFrame
    :param model: Model name
    :type model: str
    :param model_provider: Model provider
    :type model_provider: str
    :returns: Dictionary containing structured statistics
    :rtype: dict[str, Any]
    """
    import datetime

    # Calculate statistics using the same function as text report
    stats = calculate_statistics(df)

    # Get run configurations
    run_configs = []
    if not df.is_empty():
        config_df = df.group_by(
            [
                "phase_name",
                "config",
                "a2a_app_prop",
                "has_tool_augmentation",
                "has_env_events",
            ]
        ).agg(
            [
                pl.len().alias("total_runs"),
                (pl.col("success_numeric").is_not_null()).sum().alias("validated_runs"),
                (pl.col("status") == "success").sum().alias("success_runs"),
                (pl.col("status") == "failed").sum().alias("failed_runs"),
                (pl.col("status") == "exception").sum().alias("exception_runs"),
                (pl.col("status") == "no_validation").sum().alias("no_validation_runs"),
            ]
        )

        for row in config_df.iter_rows(named=True):
            run_configs.append(
                {
                    "phase_name": row["phase_name"],
                    "config": row["config"],
                    "a2a_app_prop": row["a2a_app_prop"],
                    "has_tool_augmentation": row["has_tool_augmentation"],
                    "has_env_events": row["has_env_events"],
                    "total_runs": row["total_runs"],
                    "validated_runs": row["validated_runs"],
                    "success_runs": row["success_runs"],
                    "failed_runs": row["failed_runs"],
                    "exception_runs": row["exception_runs"],
                    "no_validation_runs": row["no_validation_runs"],
                }
            )

    return {
        "metadata": {
            "model": model,
            "model_provider": model_provider,
            "timestamp": datetime.datetime.now().isoformat(),
            "report_version": "3.0",  # Updated version for cleaned up code
        },
        "statistics": stats,
        "run_configurations": run_configs,
    }


def build_trace_rows_polars(df: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Convert DataFrame to HuggingFace dataset rows format.

    Args:
        df: DataFrame with scenario results

    Returns:
        List of dictionaries representing dataset rows
    """

    rows = []

    for row in df.iter_rows(named=True):
        # Load trace data if available
        data = None
        if row["export_path"]:
            try:
                with open(row["export_path"], "r") as f:
                    data = f.read()
            except Exception as e:
                logger.warning(f"Failed to read trace file {row['export_path']}: {e}")
                data = None

        # Convert success to score
        score = row["success_numeric"]

        # Build the row
        hf_row = {
            "scenario_id": row["base_scenario_id"],
            "run_number": row["run_number"],
            "task_id": row["base_scenario_id"],  # Could add run suffix if needed
            "score": score,
            "status": row["status"],
            "data": data,
            "has_exception": row["has_exception"],
            "exception_type": row["exception_type"],
            "exception_message": row["exception_message"],
            "rationale": row["rationale"],
            "config": row["config"],
            "phase_name": row["phase_name"],
            "a2a_app_prop": float(row["a2a_app_prop"]),  # Convert string back to float
            "has_app_noise": row["has_tool_augmentation"]
            == "True",  # Convert string back to bool
            "has_env_noise": row["has_env_events"]
            == "True",  # Convert string back to bool
            # Could add run_config JSON here if needed
        }

        # Remove None values to keep the dataset clean
        hf_row = {k: v for k, v in hf_row.items() if v is not None}
        rows.append(hf_row)

    return rows
