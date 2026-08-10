# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Prompt templates for translation, review, post-editing, and oracle args."""

from __future__ import annotations


TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert translator. You translate text accurately and naturally "
    "from {src_lang} to {tgt_lang}. Respond ONLY with the translated text, "
    "nothing else."
)

TRANSLATION_PROMPT = """\
Translate the following text from {src_lang} to {tgt_lang}.
{glossary_section}
### Text to translate:
{src_text}

Keep any proper nouns, person names, IDs, email addresses, URLs, file names, and technical identifiers unchanged.
When the text contains quoted references to app content (conversation titles, email subjects, event titles), use the exact translations from the terminology section above if provided.

Respond with ONLY the translated text, no explanations or comments."""

TRANSLATION_REVIEW_SYSTEM_PROMPT = (
    "You are an expert bilingual reviewer. You will be given an original English "
    "text and its translation. You must evaluate the translation quality and "
    "respond ONLY with valid JSON."
)

TRANSLATION_REVIEW_PROMPT = """\
Review the following translation from {src_lang} to {tgt_lang}.

### Original ({src_lang}):
{src_text}

### Translation ({tgt_lang}):
{translated_text}

Evaluate the translation and respond with a JSON object containing these keys IN THIS ORDER:
- "reasoning": step-by-step analysis of the translation — compare key phrases, check for omissions, assess naturalness in the target language
- "quality": one of "good", "acceptable", "poor"
- "preserves_meaning": true/false
- "is_fluent": true/false
- "issues": list of strings describing any issues (empty list if none)
- "suggestion": improved translation if quality is not "good", otherwise null

Respond ONLY with the JSON object, no additional text."""

EXPECTED_RESPONSE_REVIEW_SYSTEM_PROMPT = (
    "You are an expert evaluator for multilingual benchmarks. You will be given "
    "an original English prompt, its translation, and the expected response for "
    "the benchmark. You must determine whether the expected response needs to be "
    "translated. Respond ONLY with valid JSON."
)

EXPECTED_RESPONSE_REVIEW_PROMPT = """\
Given a benchmark scenario where the user prompt has been translated from {src_lang} to {tgt_lang}, analyze whether the expected response also needs translation.

### Original prompt ({src_lang}):
{src_text}

### Translated prompt ({tgt_lang}):
{translated_text}

### Expected response:
{expected_response}

Respond with a JSON object containing these keys IN THIS ORDER:
- "reasoning": step-by-step analysis — is the expected response a number, a proper noun, or natural language? Would a user reading the translated prompt expect the response in the target language?
- "needs_translation": true/false — does the expected response contain natural language that should be translated?
- "reason": one of "numeric_or_factual" (numbers, dates, quantities — no translation needed), "proper_noun" (names, places — usually no translation needed), "natural_language" (contains words/phrases that should be in the target language), "mixed" (partially needs translation)
- "translation_priority": one of "none", "low", "high"
- "suggested_translated_response": if needs_translation is true, provide the translated expected response, otherwise null

Respond ONLY with the JSON object, no additional text."""

POST_EDIT_SYSTEM_PROMPT = (
    "You are an expert bilingual post-editor. Your job is to fix specific issues "
    "in a translation while preserving everything that is already correct. You "
    "respond ONLY with the corrected translation, nothing else."
)

POST_EDIT_PROMPT = """\
Fix the following translation from {src_lang} to {tgt_lang}. A reviewer flagged specific issues that need correction. Fix ONLY those issues — do not rewrite parts of the translation that are already correct.

### Original ({src_lang}):
{src_text}

### Current translation ({tgt_lang}):
{translated_text}

### Issues to fix:
{issues}

Respond with ONLY the corrected translation, no explanations or comments."""

RESPONSE_TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert translator. You will translate a short expected response "
    "from a benchmark scenario. The response must be translated in the context of "
    "the original prompt so you understand what it refers to. Respond ONLY with "
    "the translated response, nothing else."
)

RESPONSE_TRANSLATION_PROMPT = """\
Translate the following expected response from {src_lang} to {tgt_lang}.

The response was the answer to this prompt (provided for context only — do NOT translate the prompt):
### Prompt:
{prompt_context}

### Response to translate:
{src_text}

Respond with ONLY the translated response, no explanations or comments."""

# ─────────────────────────────────────────────────────────────────────────────
# Oracle arg translation and review prompt templates
# ─────────────────────────────────────────────────────────────────────────────

ORACLE_ARG_TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert translator specializing in multilingual localization of "
    "AI agent benchmarks. You translate specific tool call arguments while "
    "preserving their functional correctness in context. Respond ONLY with "
    "valid JSON."
)

ORACLE_ARG_TRANSLATION_PROMPT = """\
Translate the following tool call arguments from {src_lang} to {tgt_lang}.

These arguments belong to the oracle (ground-truth) tool call chain of a GAIA2 benchmark scenario.

The user's original request was:
### User prompt:
{user_prompt}
{glossary_section}
### Arguments to translate:
Each entry below has a "context" field showing which tool call the argument belongs to, and a "text" field with the value to translate.

{args_to_translate}

Translate each "text" value naturally into {tgt_lang}, preserving meaning and intent.
Use the "context" field to understand what kind of text it is (e.g. email body, search query, event title).
Keep any proper nouns, IDs, email addresses, URLs, or technical identifiers unchanged.
When a value matches a term from the terminology section above, use the exact translation provided.

Respond with a JSON object mapping each numeric key to its translated text:
{{
  "0": "translated value for arg 0",
  "1": "translated value for arg 1"
}}

The keys MUST match the numeric keys provided above exactly. Respond ONLY with the JSON object."""

ORACLE_ARG_REVIEW_SYSTEM_PROMPT = (
    "You are an expert bilingual reviewer specializing in multilingual localization "
    "of AI agent benchmarks. You evaluate translations of tool call arguments "
    "in their original scenario context — the user prompt, the tool call chain, "
    "and all sibling arguments — to assess faithfulness and consistency. "
    "Respond ONLY with valid JSON."
)

ORACLE_ARG_REVIEW_PROMPT = """\
Review the following translated tool call arguments from {src_lang} to {tgt_lang}.

These arguments belong to the oracle (ground-truth) tool call chain of a GAIA2 benchmark scenario.

The user's original request was:
### User prompt:
{user_prompt}

### Arguments to review:
Each entry below has a "context" field showing which tool call the argument belongs to, an "original" field with the source text, and a "translation" field with the translated text.

{args_to_review}

For each argument, evaluate whether the translation:
1. Preserves the meaning and intent of the original in the context of the user prompt and tool call
2. Is natural and fluent in {tgt_lang}
3. Correctly preserves proper nouns, IDs, email addresses, URLs, and technical identifiers
4. Is consistent with the other translated arguments in the same scenario

Respond with a JSON object mapping each numeric key to its review:
{{
  "0": {{"reasoning": "...", "quality": "good|acceptable|poor", "preserves_meaning": true/false, "is_fluent": true/false, "issues": [], "suggestion": null}},
  "1": {{"reasoning": "...", "quality": "poor", "preserves_meaning": false, "is_fluent": true, "issues": ["issue description"], "suggestion": "improved translation"}}
}}

The keys MUST match the numeric keys provided above exactly. Respond ONLY with the JSON object."""
