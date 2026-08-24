# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Constants for the GAIA2 translation pipeline."""

from __future__ import annotations


ALL_SUBSETS = ["search", "execution", "ambiguity", "adaptability"]

TRANSLATABLE_ARG_NAMES: set[str] = {
    "content",
    "body",
    "subject",
    "message",
    "description",
    "title",
    "text",
    "query",
    "note",
    "comment",
}

# Skip user prompt events — these have a dedicated reviewed translation pipeline.
# Only send_message_to_agent is skipped; send_message_to_user oracle events
# (expected responses) are translated via the oracle args pipeline alongside
# other tool-call args, since ALL responses should be localized.
SKIP_FUNCTIONS: set[str] = {
    "send_message_to_agent",
}

LANG_CODE_TO_NAME: dict[str, str] = {
    # Source language
    "eng_Latn": "English",
    # ── Tier 0 ────────────────────────────────────────────────────────────
    "spa_Latn": "Spanish",
    "por_Latn": "Brazilian Portuguese",
    "hin_Deva": "Hindi",
    # ── Tier 1 ────────────────────────────────────────────────────────────
    "ind_Latn": "Indonesian",
    "ara_Arab": "Arabic",
    "tur_Latn": "Turkish",
    "ben_Beng": "Bengali",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "jpn_Jpan": "Japanese",
    "mar_Deva": "Marathi",
    "guj_Gujr": "Gujarati",
    "kan_Knda": "Kannada",
    # ── Tier 2 ────────────────────────────────────────────────────────────
    "tgl_Latn": "Filipino",
    "vie_Latn": "Vietnamese",
    "rus_Cyrl": "Russian",
    "tha_Thai": "Thai",
    "urd_Arab": "Urdu",
    "jav_Latn": "Javanese",
    "ita_Latn": "Italian",
    "pol_Latn": "Polish",
    "tam_Taml": "Tamil",
    "mal_Mlym": "Malayalam",
    "tel_Telu": "Telugu",
    "cmn_Hans": "Chinese (Simplified)",
    "kor_Hang": "Korean",
    "cmn_Hant": "Chinese (Traditional)",
    # ── Romanized variants ────────────────────────────────────────────────
    "ben_Latn": "Bengali (Romanized)",
    "tam_Latn": "Tamil (Romanized)",
    "tel_Latn": "Telugu (Romanized)",
    "mar_Latn": "Marathi (Romanized)",
    "kan_Latn": "Kannada (Romanized)",
    "guj_Latn": "Gujarati (Romanized)",
    "urd_Latn": "Urdu (Romanized)",
    "ara_Latn": "Arabic (Romanized)",
    # ── English dialects ──────────────────────────────────────────────────
    "eng_Latn_IN": "Indian English",
    "eng_Latn_GB": "British English",
    "eng_Latn_AU": "Australian English",
    # ── Code-switched ─────────────────────────────────────────────────────
    "hin_Latn_CS": "Hinglish",
}

# ─────────────────────────────────────────────────────────────────────────────
# Language type classification
# ─────────────────────────────────────────────────────────────────────────────

ROMANIZED_LANGUAGES: frozenset[str] = frozenset(
    {
        "ben_Latn",
        "tam_Latn",
        "tel_Latn",
        "mar_Latn",
        "kan_Latn",
        "guj_Latn",
        "urd_Latn",
        "ara_Latn",
    }
)

DIALECT_LANGUAGES: frozenset[str] = frozenset(
    {
        "eng_Latn_IN",
        "eng_Latn_GB",
        "eng_Latn_AU",
    }
)

CODE_SWITCHED_LANGUAGES: frozenset[str] = frozenset(
    {
        "hin_Latn_CS",
    }
)

# Languages where GlotLID cannot reliably verify the output script or dialect.
LID_SKIP_LANGUAGES: frozenset[str] = (
    ROMANIZED_LANGUAGES | DIALECT_LANGUAGES | CODE_SWITCHED_LANGUAGES
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt-language helpers
# ─────────────────────────────────────────────────────────────────────────────

# Base language name for romanized codes (strip " (Romanized)" suffix).
_ROMANIZED_BASE_NAMES: dict[str, str] = {
    code: LANG_CODE_TO_NAME[code].replace(" (Romanized)", "")
    for code in ROMANIZED_LANGUAGES
}

# Per-dialect adaptation instructions injected into translation prompts.
_DIALECT_INSTRUCTIONS: dict[str, str] = {
    "eng_Latn_IN": (
        "\n### Adaptation Instructions:\n"
        "This is a dialect adaptation, not a translation to a different language. "
        "Adapt the American English text to Indian English conventions:\n"
        "- Use Indian English spelling and vocabulary where applicable\n"
        "- Adapt cultural references, units, and date formats to Indian conventions\n"
        "- Preserve the meaning and intent exactly\n"
        "- Use expressions and phrasing natural to Indian English speakers\n"
    ),
    "eng_Latn_GB": (
        "\n### Adaptation Instructions:\n"
        "This is a dialect adaptation, not a translation to a different language. "
        "Adapt the American English text to British English conventions:\n"
        "- Use British spelling (colour, analyse, centre, etc.)\n"
        "- Use British vocabulary (flat, boot, lorry, etc.)\n"
        "- Adapt cultural references, units, and date formats to British conventions\n"
        "- Preserve the meaning and intent exactly\n"
    ),
    "eng_Latn_AU": (
        "\n### Adaptation Instructions:\n"
        "This is a dialect adaptation, not a translation to a different language. "
        "Adapt the American English text to Australian English conventions:\n"
        "- Use Australian English spelling and vocabulary where applicable\n"
        "- Adapt cultural references, units, and date formats to Australian conventions\n"
        "- Preserve the meaning and intent exactly\n"
        "- Use expressions and phrasing natural to Australian English speakers\n"
    ),
}


def get_language_display_name(lang_code: str) -> str:
    """Return a human-readable language name for use in LLM prompts."""
    return LANG_CODE_TO_NAME.get(lang_code, lang_code)


def get_special_instructions(lang_code: str) -> str:
    """Return extra prompt instructions for non-standard language types.

    Returns an empty string for standard languages, or a block of instructions
    for romanized, dialect, or code-switched targets.  The returned text is
    designed to be prepended to the ``glossary_section`` in translation prompts.
    """
    if lang_code in ROMANIZED_LANGUAGES:
        base = _ROMANIZED_BASE_NAMES[lang_code]
        return (
            f"\n### Script Instructions:\n"
            f"IMPORTANT: Write the {base} text using Latin/Roman script "
            f"(romanized {base}), NOT the native {base} script. "
            f"The output must be readable by someone who cannot read the native "
            f"{base} script but can read Latin letters.\n"
        )

    if lang_code in DIALECT_LANGUAGES:
        return _DIALECT_INSTRUCTIONS.get(lang_code, "")

    if lang_code in CODE_SWITCHED_LANGUAGES:
        return (
            "\n### Code-Switching Instructions:\n"
            "IMPORTANT: Produce Hinglish output — naturally code-switched "
            "Hindi-English text written entirely in Latin/Roman script. "
            "Guidelines:\n"
            "- Mix Hindi and English words and phrases naturally, as a bilingual "
            "Hindi-English speaker would in casual conversation\n"
            "- Use Latin/Roman script throughout (no Devanagari)\n"
            "- English technical terms, proper nouns, and common English words "
            "should stay in English\n"
            "- Hindi words should be transliterated into Latin script\n"
            "- The mixing should feel natural and conversational, not forced\n"
        )

    return ""
