# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import sys
import types
from pathlib import Path

import click
import pytest
from gaia2_runner import hf_dataset
from gaia2_runner.config import CANONICAL_SPLITS, MULTILINGUAL_SPLITS

_OMNILINGUAL = "facebook/omnilingual-gaia2"
_ENGLISH = "meta-agents-research-environments/gaia2-cli"


class _FakeLoader:
    """Records the (dataset_id, config_name, split) triples it is asked for."""

    def __init__(self, rows: list[dict[str, str]] | None = None) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self._rows = (
            rows if rows is not None else [{"scenario_id": "s1", "scenario": "{}"}]
        )

    def __call__(self, dataset_id, config_name, split=None, token=None):
        self.calls.append((dataset_id, config_name, split))
        return self._rows

    @property
    def config_names(self) -> list[str]:
        return [config_name for _, config_name, _ in self.calls]


def _install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loader: object,
) -> Path:
    """Point the cache at tmp_path and stub the `datasets` import.

    `_CACHE_DIR` is read at import time, so setting $GAIA2_HF_CACHE here would
    have no effect; and `load_dataset` is imported inside the function, which is
    what lets us stub the module without importing the real library.
    """
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(hf_dataset, "_CACHE_DIR", cache_root)
    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=loader)
    )
    return cache_root


def test_language_builds_prefixed_config_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _FakeLoader()
    _install(monkeypatch, tmp_path, loader)

    hf_dataset.download_hf_dataset(_OMNILINGUAL, splits=["search"], language="spa_Latn")

    assert loader.calls == [(_OMNILINGUAL, "spa_Latn_search", "test")]


def test_language_materializes_bare_split_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _FakeLoader([{"scenario_id": "abc", "scenario": '{"events": []}'}])
    _install(monkeypatch, tmp_path, loader)

    cache_dir = Path(
        hf_dataset.download_hf_dataset(
            _OMNILINGUAL, splits=["search"], language="spa_Latn"
        )
    )

    # The directory is the bare split name even though the HF config was
    # prefixed: _infer_result_split() reads the split from parts[0].
    assert cache_dir.name == "facebook_omnilingual-gaia2_spa_Latn"
    scenario_file = cache_dir / "search" / "abc.json"
    assert scenario_file.read_text() == '{"events": []}'


def test_no_language_preserves_legacy_cache_dir_and_config_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _FakeLoader()
    _install(monkeypatch, tmp_path, loader)

    cache_dir = Path(hf_dataset.download_hf_dataset(_ENGLISH, splits=["search"]))

    # Backward-compat guard: existing populated caches must stay valid.
    assert cache_dir.name == "meta-agents-research-environments_gaia2-cli"
    assert loader.config_names == ["search"]


def test_language_defaults_to_multilingual_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _FakeLoader()
    _install(monkeypatch, tmp_path, loader)

    hf_dataset.download_hf_dataset(_OMNILINGUAL, language="spa_Latn")

    assert loader.config_names == [f"spa_Latn_{s}" for s in MULTILINGUAL_SPLITS]
    assert not any(name.endswith("_time") for name in loader.config_names)


def test_language_all_expands_to_multilingual_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _FakeLoader()
    _install(monkeypatch, tmp_path, loader)

    hf_dataset.download_hf_dataset(_OMNILINGUAL, splits=["all"], language="spa_Latn")

    assert loader.config_names == [f"spa_Latn_{s}" for s in MULTILINGUAL_SPLITS]


def test_no_language_defaults_to_canonical_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _FakeLoader()
    _install(monkeypatch, tmp_path, loader)

    hf_dataset.download_hf_dataset(_ENGLISH)

    assert loader.config_names == list(CANONICAL_SPLITS)
    assert "time" in loader.config_names


def test_languages_do_not_share_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _FakeLoader()
    _install(monkeypatch, tmp_path, loader)

    spa = hf_dataset.download_hf_dataset(
        _OMNILINGUAL, splits=["search"], language="spa_Latn"
    )
    deu = hf_dataset.download_hf_dataset(
        _OMNILINGUAL, splits=["search"], language="deu_Latn"
    )

    assert spa != deu
    assert (Path(spa) / "search").is_dir()
    assert (Path(deu) / "search").is_dir()
    assert loader.config_names == ["spa_Latn_search", "deu_Latn_search"]


def test_cached_splits_skip_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exploding_loader(*args, **kwargs):
        raise AssertionError("load_dataset must not be called for a cached split")

    cache_root = _install(monkeypatch, tmp_path, exploding_loader)
    cached = cache_root / "facebook_omnilingual-gaia2_spa_Latn" / "search"
    cached.mkdir(parents=True)
    (cached / "already.json").write_text("{}")

    result = hf_dataset.download_hf_dataset(
        _OMNILINGUAL, splits=["search"], language="spa_Latn"
    )

    assert result == str(cached.parent)


def test_failed_config_is_not_left_as_empty_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_loader(*args, **kwargs):
        raise ValueError("BuilderConfig 'spa_search' not found")

    cache_root = _install(monkeypatch, tmp_path, failing_loader)

    with pytest.raises(click.UsageError) as excinfo:
        hf_dataset.download_hf_dataset(_OMNILINGUAL, splits=["search"], language="spa")

    # An empty split dir left behind would pass the next run's missing-split
    # check and silently resolve zero scenarios.
    assert not (cache_root / "facebook_omnilingual-gaia2_spa" / "search").exists()
    assert "spa_search" in str(excinfo.value)
    assert "--language" in str(excinfo.value)


def test_unknown_config_error_lists_available_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_loader(*args, **kwargs):
        raise ValueError("not found")

    _install(monkeypatch, tmp_path, failing_loader)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            get_dataset_config_names=lambda dataset_id, token=None: [
                "spa_Latn_search",
                "deu_Latn_search",
            ]
        ),
    )

    with pytest.raises(click.UsageError) as excinfo:
        hf_dataset.download_hf_dataset(_OMNILINGUAL, splits=["search"], language="spa")

    assert "spa_Latn_search" in str(excinfo.value)


def test_config_listing_failure_does_not_mask_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_loader(*args, **kwargs):
        raise ValueError("the original problem")

    def exploding_lister(dataset_id, token=None):
        raise OSError("no network")

    _install(monkeypatch, tmp_path, failing_loader)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(get_dataset_config_names=exploding_lister),
    )

    with pytest.raises(click.UsageError) as excinfo:
        hf_dataset.download_hf_dataset(
            _OMNILINGUAL, splits=["search"], language="spa_Latn"
        )

    assert "the original problem" in str(excinfo.value)
    assert "Available configs" not in str(excinfo.value)


def test_interrupt_mid_download_does_not_leave_partial_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted_loader(*args, **kwargs):
        raise KeyboardInterrupt

    cache_root = _install(monkeypatch, tmp_path, interrupted_loader)

    # The interrupt must propagate untouched, not be wrapped in a UsageError...
    with pytest.raises(KeyboardInterrupt):
        hf_dataset.download_hf_dataset(
            _OMNILINGUAL, splits=["search"], language="spa_Latn"
        )

    # ...and must not leave a partial directory that the cache check would
    # accept on the next run.
    assert not (cache_root / "facebook_omnilingual-gaia2_spa_Latn" / "search").exists()
