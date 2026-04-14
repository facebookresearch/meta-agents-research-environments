# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for recursive scenario discovery and mirrored output structure.

Covers:
1. _load_dataset_scenarios — recursive vs non-recursive discovery
2. _effective_output_dir — output path mirroring logic
"""

from __future__ import annotations

import json
from pathlib import Path

from gaia2_runner.cli import (
    _effective_output_dir,
    _load_dataset_scenarios,
    _load_subset_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario(path: Path, scenario_id: str = "test_scenario") -> None:
    """Write a minimal scenario JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {"definition": {"scenario_id": scenario_id}},
                "events": [],
            }
        )
    )


# ---------------------------------------------------------------------------
# Tests for _load_dataset_scenarios
# ---------------------------------------------------------------------------


class TestLoadDatasetScenarios:
    """Test scenario discovery from dataset directories."""

    def test_flat_directory_recursive(self, tmp_path: Path) -> None:
        """Recursive search finds scenarios in a flat directory."""
        _make_scenario(tmp_path / "s1.json", "s1")
        _make_scenario(tmp_path / "s2.json", "s2")

        paths, root, tmpdir = _load_dataset_scenarios(
            str(tmp_path), None, recursive=True
        )

        assert len(paths) == 2
        assert root == tmp_path
        assert tmpdir is None
        assert {p.name for p in paths} == {"s1.json", "s2.json"}

    def test_flat_directory_non_recursive(self, tmp_path: Path) -> None:
        """Non-recursive search finds scenarios in a flat directory."""
        _make_scenario(tmp_path / "s1.json", "s1")
        _make_scenario(tmp_path / "s2.json", "s2")

        paths, root, tmpdir = _load_dataset_scenarios(
            str(tmp_path), None, recursive=False
        )

        assert len(paths) == 2
        assert root == tmp_path
        assert tmpdir is None

    def test_nested_directory_recursive(self, tmp_path: Path) -> None:
        """Recursive search finds scenarios in nested subdirectories."""
        _make_scenario(tmp_path / "search" / "validation" / "s1.json", "s1")
        _make_scenario(tmp_path / "time" / "validation" / "s2.json", "s2")
        _make_scenario(tmp_path / "top_level.json", "top")

        paths, root, tmpdir = _load_dataset_scenarios(
            str(tmp_path), None, recursive=True
        )

        assert len(paths) == 3
        assert root == tmp_path
        names = {p.name for p in paths}
        assert names == {"s1.json", "s2.json", "top_level.json"}

    def test_nested_directory_non_recursive(self, tmp_path: Path) -> None:
        """Non-recursive search only finds top-level scenarios."""
        _make_scenario(tmp_path / "search" / "validation" / "s1.json", "s1")
        _make_scenario(tmp_path / "top_level.json", "top")

        paths, root, tmpdir = _load_dataset_scenarios(
            str(tmp_path), None, recursive=False
        )

        assert len(paths) == 1
        assert paths[0].name == "top_level.json"

    def test_limit_applied(self, tmp_path: Path) -> None:
        """Limit is applied after collecting all scenarios."""
        for i in range(5):
            _make_scenario(tmp_path / f"s{i}.json", f"s{i}")

        paths, _, _ = _load_dataset_scenarios(str(tmp_path), 3, recursive=True)

        assert len(paths) == 3

    def test_jsonl_input(self, tmp_path: Path) -> None:
        """JSONL input expands to individual scenario files."""
        jsonl_path = tmp_path / "scenarios.jsonl"
        scenarios = [
            {"metadata": {"definition": {"scenario_id": f"s{i}"}}, "events": []}
            for i in range(3)
        ]
        jsonl_path.write_text("\n".join(json.dumps(s) for s in scenarios))

        paths, root, tmpdir = _load_dataset_scenarios(
            str(jsonl_path), None, recursive=True
        )

        assert len(paths) == 3
        assert root is None  # no dataset root for JSONL
        assert tmpdir is not None

        # Clean up tmpdir
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_single_file_input(self, tmp_path: Path) -> None:
        """Single file input returns just that file."""
        scenario = tmp_path / "single.json"
        _make_scenario(scenario, "single")

        paths, root, tmpdir = _load_dataset_scenarios(
            str(scenario), None, recursive=True
        )

        assert len(paths) == 1
        assert paths[0] == scenario
        assert root is None
        assert tmpdir is None

    def test_default_is_recursive(self, tmp_path: Path) -> None:
        """Default behavior (no recursive kwarg) is recursive."""
        _make_scenario(tmp_path / "sub" / "s1.json", "s1")

        paths, _, _ = _load_dataset_scenarios(str(tmp_path), None)

        assert len(paths) == 1
        assert paths[0].name == "s1.json"

    def test_recursive_ignores_hidden_metadata_directories(
        self, tmp_path: Path
    ) -> None:
        """Hidden directories such as HF cache metadata are skipped."""
        _make_scenario(tmp_path / "search" / "s1.json", "s1")
        hidden = tmp_path / ".cache" / "huggingface" / "download.json"
        hidden.parent.mkdir(parents=True, exist_ok=True)
        hidden.write_text(json.dumps({"foo": "bar"}))

        paths, _, _ = _load_dataset_scenarios(str(tmp_path), None)

        assert len(paths) == 1
        assert paths[0].name == "s1.json"

    def test_recursive_ignores_non_scenario_json_files(self, tmp_path: Path) -> None:
        """Non-scenario JSON files should not be treated as runnable scenarios."""
        _make_scenario(tmp_path / "search" / "s1.json", "s1")
        (tmp_path / "manifest.json").write_text(json.dumps({"name": "dataset"}))

        paths, _, _ = _load_dataset_scenarios(str(tmp_path), None)

        assert len(paths) == 1
        assert paths[0].name == "s1.json"


# ---------------------------------------------------------------------------
# Tests for _effective_output_dir
# ---------------------------------------------------------------------------


class TestEffectiveOutputDir:
    """Test per-scenario output directory computation."""

    def test_no_output_dir(self) -> None:
        """Returns None when output_dir is None."""
        result = _effective_output_dir(
            None, Path("/data/dataset/s1.json"), Path("/data/dataset")
        )
        assert result is None

    def test_no_dataset_root(self) -> None:
        """Returns output_dir unchanged when dataset_root is None (JSONL)."""
        result = _effective_output_dir(
            "/tmp/output", Path("/tmp/scenarios/s1.json"), None
        )
        assert result == "/tmp/output"

    def test_flat_scenario(self, tmp_path: Path) -> None:
        """Scenario in dataset root returns output_dir unchanged."""
        dataset_root = tmp_path / "dataset"
        scenario = dataset_root / "s1.json"

        result = _effective_output_dir("/tmp/output", scenario, dataset_root)
        assert result == "/tmp/output"

    def test_nested_scenario(self) -> None:
        """Scenario in subdirectory includes relative subdir in output."""
        result = _effective_output_dir(
            "/tmp/output",
            Path("/data/dataset/search/validation/s1.json"),
            Path("/data/dataset"),
        )
        assert result == "/tmp/output/search/validation"

    def test_deeply_nested_scenario(self) -> None:
        """Deeply nested scenario preserves full relative path."""
        result = _effective_output_dir(
            "/tmp/output",
            Path("/data/dataset/a/b/c/s1.json"),
            Path("/data/dataset"),
        )
        assert result == "/tmp/output/a/b/c"

    def test_single_level_nested(self) -> None:
        """Single level nesting works correctly."""
        result = _effective_output_dir(
            "/tmp/output",
            Path("/data/dataset/search/s1.json"),
            Path("/data/dataset"),
        )
        assert result == "/tmp/output/search"


# ---------------------------------------------------------------------------
# Tests for subset filtering
# ---------------------------------------------------------------------------


def _make_subset_manifest(path: Path, splits: dict[str, list[str]]) -> None:
    """Write a subset manifest JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "test-subset",
                "description": "Test subset",
                "version": "1.0",
                "source_dataset": "test",
                "selection": {"method": "test"},
                "splits": splits,
            }
        )
    )


class TestLoadSubsetIds:
    """Test _load_subset_ids manifest parsing."""

    def test_loads_all_splits(self, tmp_path: Path) -> None:
        manifest = tmp_path / "subset.json"
        _make_subset_manifest(
            manifest,
            {"execution": ["s1", "s2"], "search": ["s3"]},
        )
        ids = _load_subset_ids(str(manifest))
        assert ids == {"s1", "s2", "s3"}

    def test_empty_splits(self, tmp_path: Path) -> None:
        manifest = tmp_path / "subset.json"
        _make_subset_manifest(manifest, {})
        ids = _load_subset_ids(str(manifest))
        assert ids == set()


class TestSubsetFiltering:
    """Test _load_dataset_scenarios with --subset filtering."""

    def test_filters_to_subset(self, tmp_path: Path) -> None:
        """Only scenarios in the subset manifest are returned."""
        for sid in ["s1", "s2", "s3", "s4", "s5"]:
            _make_scenario(tmp_path / f"{sid}.json", sid)

        manifest = tmp_path / "subset.json"
        _make_subset_manifest(manifest, {"split": ["s2", "s4"]})

        paths, _, _ = _load_dataset_scenarios(str(tmp_path), None, subset=str(manifest))

        assert len(paths) == 2
        ids = {p.stem for p in paths}
        assert ids == {"s2", "s4"}

    def test_subset_with_limit(self, tmp_path: Path) -> None:
        """Limit is applied after subset filtering."""
        for sid in ["s1", "s2", "s3", "s4"]:
            _make_scenario(tmp_path / f"{sid}.json", sid)

        manifest = tmp_path / "subset.json"
        _make_subset_manifest(manifest, {"split": ["s1", "s2", "s3"]})

        paths, _, _ = _load_dataset_scenarios(str(tmp_path), 2, subset=str(manifest))

        assert len(paths) == 2

    def test_subset_no_match(self, tmp_path: Path) -> None:
        """Returns empty list when no scenarios match."""
        _make_scenario(tmp_path / "s1.json", "s1")

        manifest = tmp_path / "subset.json"
        _make_subset_manifest(manifest, {"split": ["nonexistent"]})

        paths, _, _ = _load_dataset_scenarios(str(tmp_path), None, subset=str(manifest))

        assert len(paths) == 0

    def test_no_subset_returns_all(self, tmp_path: Path) -> None:
        """Without subset, all scenarios are returned."""
        for sid in ["s1", "s2", "s3"]:
            _make_scenario(tmp_path / f"{sid}.json", sid)

        paths, _, _ = _load_dataset_scenarios(str(tmp_path), None, subset=None)

        assert len(paths) == 3

    def test_subset_across_splits(self, tmp_path: Path) -> None:
        """Subset IDs from multiple splits are merged."""
        (tmp_path / "exec").mkdir()
        (tmp_path / "search").mkdir()
        _make_scenario(tmp_path / "exec" / "s1.json", "s1")
        _make_scenario(tmp_path / "exec" / "s2.json", "s2")
        _make_scenario(tmp_path / "search" / "s3.json", "s3")
        _make_scenario(tmp_path / "search" / "s4.json", "s4")

        manifest = tmp_path / "subset.json"
        _make_subset_manifest(manifest, {"execution": ["s1"], "search": ["s3"]})

        paths, _, _ = _load_dataset_scenarios(
            str(tmp_path), None, recursive=True, subset=str(manifest)
        )

        assert len(paths) == 2
        ids = {p.stem for p in paths}
        assert ids == {"s1", "s3"}
