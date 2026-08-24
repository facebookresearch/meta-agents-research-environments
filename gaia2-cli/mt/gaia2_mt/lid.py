# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Lightweight Language Identification using GlotLID (fastText-based, CPU-only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gaia2_mt.data import LID_SKIP_LANGUAGES, SplitResult


logger: logging.Logger = logging.getLogger(__name__)

# GlotLID uses ISO 639-3 + script codes.  Where Omnilingual-GAIA2 uses a different
# code, map to the GlotLID equivalent.
# Note: GlotLID uses cmn_Hani for both Simplified and Traditional Chinese.
_GAIA2_MT_TO_GLOTLID: dict[str, str] = {
    "cmn_Hans": "cmn_Hani",
    "cmn_Hant": "cmn_Hani",
    # Korean: GlotLID uses kor_Hang but some builds use kor_Kore
    "kor_Hang": "kor_Hang",
    # Javanese: GlotLID may label as jav_Latn
    "jav_Latn": "jav_Latn",
}

_GLOTLID_TO_GAIA2_MT: dict[str, set[str]] = {}
for _og, _gl in _GAIA2_MT_TO_GLOTLID.items():
    _GLOTLID_TO_GAIA2_MT.setdefault(_gl, set()).add(_og)

_MIN_TEXT_LENGTH = 10


@dataclass
class LidResult:
    """Result of a single LID check."""

    detected_lang: str
    confidence: float
    expected_lang: str
    is_correct: bool


@dataclass
class LidCategoryStats:
    """Aggregated LID statistics for one category of translated content."""

    total: int = 0
    checked: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class LidReport:
    """Full LID validation report for a SplitResult."""

    prompt_stats: LidCategoryStats = field(default_factory=LidCategoryStats)
    oracle_arg_stats: LidCategoryStats = field(default_factory=LidCategoryStats)
    app_state_stats: LidCategoryStats = field(default_factory=LidCategoryStats)
    prompt_results: list[LidResult | None] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)


_model = None


def _get_model():  # noqa: ANN202
    global _model
    if _model is None:
        import fasttext
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        logger.info(f"Loading GlotLID model from {model_path}")
        _model = fasttext.load_model(model_path)
    return _model


def detect(text: str) -> tuple[str, float]:
    """Detect the language of *text*.

    Returns ``(lang_code, confidence)`` where *lang_code* uses GlotLID's
    native ``{iso639-3}_{script}`` format.
    """
    model = _get_model()
    labels, scores = model.predict(text.replace("\n", " "), k=1)
    lang_code = labels[0].replace("__label__", "")
    return lang_code, float(scores[0])


def check_language(
    text: str,
    expected_lang: str,
    threshold: float = 0.3,
) -> LidResult:
    """Check whether *text* is in the expected language.

    The *expected_lang* should be an Omnilingual-GAIA2 code (e.g. ``cmn_Hans``).
    Mapping to GlotLID codes is handled internally.
    """
    glotlid_expected = _GAIA2_MT_TO_GLOTLID.get(expected_lang, expected_lang)
    detected_lang, confidence = detect(text)

    is_correct = (detected_lang == glotlid_expected) and (confidence >= threshold)
    # Map detected lang back to Omnilingual-GAIA2 code for display.
    # For many-to-one mappings (e.g. cmn_Hani → cmn_Hans/cmn_Hant),
    # show the expected code if it matches, otherwise the raw GlotLID code.
    reverse = _GLOTLID_TO_GAIA2_MT.get(detected_lang, set())
    if expected_lang in reverse:
        display_lang = expected_lang
    elif len(reverse) == 1:
        display_lang = next(iter(reverse))
    else:
        display_lang = detected_lang

    return LidResult(
        detected_lang=display_lang,
        confidence=confidence,
        expected_lang=expected_lang,
        is_correct=is_correct,
    )


def _check_texts(
    texts: list[str | None],
    expected_lang: str,
    threshold: float,
    category: str,
    stats: LidCategoryStats,
    failures: list[dict],
    results_out: list[LidResult | None] | None = None,
) -> None:
    """Check a list of texts and update *stats* / *failures* in place."""
    for idx, text in enumerate(texts):
        stats.total += 1
        if not text or len(text.strip()) < _MIN_TEXT_LENGTH:
            stats.skipped += 1
            if results_out is not None:
                results_out.append(None)
            continue

        stats.checked += 1
        result = check_language(text, expected_lang, threshold)

        if results_out is not None:
            results_out.append(result)

        if result.is_correct:
            stats.passed += 1
        else:
            stats.failed += 1
            failures.append(
                {
                    "category": category,
                    "index": idx,
                    "detected_lang": result.detected_lang,
                    "confidence": result.confidence,
                    "expected_lang": expected_lang,
                    "text_preview": text[:80],
                }
            )


def validate_translations_lid(
    result: SplitResult,
    tgt_lang: str,
    threshold: float = 0.3,
) -> LidReport:
    """Run GlotLID validation on all translated content in *result*.

    Returns a :class:`LidReport` with per-category statistics and a list of
    failures.  This is informational — it does **not** block the pipeline.

    For romanized variants, English dialects, and code-switched languages
    (Hinglish), LID validation is skipped entirely because GlotLID cannot
    reliably identify these output types.  An empty report is returned with
    a log message.
    """
    if tgt_lang in LID_SKIP_LANGUAGES:
        logger.info(
            f"LID check skipped for {tgt_lang}: language type not supported by GlotLID "
            f"(romanized, dialect, or code-switched)"
        )
        return LidReport()

    report = LidReport()

    # 1. Prompts
    _check_texts(
        result.translated_prompts,
        tgt_lang,
        threshold,
        category="prompt",
        stats=report.prompt_stats,
        failures=report.failures,
        results_out=report.prompt_results,
    )

    # 2. Oracle arg translations (dict values)
    oracle_values = list(result.oracle_arg_translations.values())
    _check_texts(
        oracle_values,
        tgt_lang,
        threshold,
        category="oracle_arg",
        stats=report.oracle_arg_stats,
        failures=report.failures,
    )

    # 3. App state translations (dict values)
    app_state_values = list(result.app_state_translations.values())
    _check_texts(
        app_state_values,
        tgt_lang,
        threshold,
        category="app_state",
        stats=report.app_state_stats,
        failures=report.failures,
    )

    # Log summary
    for name, stats in [
        ("prompts", report.prompt_stats),
        ("oracle_args", report.oracle_arg_stats),
        ("app_state", report.app_state_stats),
    ]:
        logger.info(
            f"LID check [{name}]: "
            f"{stats.passed}/{stats.checked} correct "
            f"({stats.skipped} skipped, {stats.failed} failed)"
        )

    if report.failures:
        logger.warning(f"LID: {len(report.failures)} texts detected as wrong language")
        for f in report.failures[:10]:
            logger.warning(
                f"  [{f['category']}#{f['index']}] "
                f"expected={f['expected_lang']} "
                f"detected={f['detected_lang']} "
                f"(conf={f['confidence']:.2f}): "
                f"{f['text_preview']!r}"
            )
        if len(report.failures) > 10:
            logger.warning(f"  ... and {len(report.failures) - 10} more failures")

    return report
