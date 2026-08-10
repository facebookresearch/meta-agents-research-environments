# OmniGAIA Judge Prompts

OmniGAIA overrides gaia2-core judge prompt system prompts to support multilingual
evaluation. The API surface (tools, parameters) stays in English, but all
scenario content — calendar events, email bodies, messages, oracle references —
is translated to the target language. The agent is expected to reply in the
target language.

## Architecture

```
gaia2_core/judge/omnigaia_prompts.py            ← raw prompt strings
gaia2_core/judge/prompt_overrides.py            ← wraps into LLMFunctionTemplates,
                                                  registers in override registry
```

The judge prompts live in `gaia2_core`, not in this package, because the judge
runs inside the runner container. This directory holds only the *translation*
prompt templates ([`translation.py`](translation.py)). No gaia2-core code
imports omnigaia.

User/assistant prompt templates and few-shot examples are reused from gaia2-core;
only system prompts are overridden.

## Versions

| Version | Key | Scope |
|---------|-----|-------|
| **default** | `"default"` | Uses gaia2-core prompts as-is (no override). |
| **omnigaia** | `"omnigaia"` | **6 checkers overridden.** gaia2-core latest + multilingual awareness. |

## What `omnigaia` Overrides

`omnigaia` starts from the latest gaia2-core prompts and applies minimal,
targeted edits to remove English-centric biases. Each change is tagged with
`# OMNIGAIA` in the source.

### Checkers overridden

| Checker | Override key | What changed |
|---------|-------------|--------------|
| **USER_MESSAGE_CHECKER** | `user_message_checker_prompt_templates` | Added **Cross-Lingual Equivalence** rule: evaluate based on semantic content regardless of language. |
| **TONE_CHECKER** | `tone_checker_prompt_templates` | `"plain English"` → `"plain text (in any language)"`; grammar check scoped to the language the text is written in. |
| **EMAIL_CHECKER** | `email_checker_prompt_templates` | Added multilingual note; generalized greeting/sign-off examples to be language-agnostic. |
| **MESSAGE_CHECKER** | `message_checker_prompt_templates` | Added multilingual note; generalized greeting/sign-off examples to be language-agnostic. |
| **EVENT_CHECKER** | `event_checker_prompt_templates` | Added multilingual note; title equivalence stated as language-independent. |
| **SIGNATURE_CHECKER** | `signature_checker_prompt_templates` | Extended placeholder detection to include translations of "Assistant", "User", "Your Name" in any language. |

### Checkers left unchanged (gaia2-core defaults)

| Checker | Why |
|---------|-----|
| **CAB_CHECKER** | Compares addresses — locale-native, language-neutral. |
| **CONTENT_CHECKER** | Generic parameter comparison — language-neutral. |
| **SANITY_CHECKER** | Detects placeholders/garbled text — language-neutral enough. |
| **SUBTASK_EXTRACTOR** | Operates on English task descriptions from the benchmark. |

## Usage

Set the `JUDGE_PROMPT_VERSION` variable in your shell script:

```bash
JUDGE_PROMPT_VERSION="omnigaia"
```

This is looked up in `OMNIGAIA_JUDGE_PROMPT_TEMPLATE_OVERRIDES_REGISTRY` and
passed to `GraphPerEventJudgeConfig(judge_prompt_template_overrides=...)`.

## When to Update

Create a new version (`omnigaia_v3`, etc.) when:
- gaia2-core updates its checker prompts with changes you want to pick up
  (rebase on gaia2-core latest and re-apply the multilingual edits).
- You need additional omnigaia-specific evaluation rules beyond multilinguality.

## Historical Context

### v0 (strict completeness)
Added on top of a now-outdated gaia2-core baseline:
- Cross-Lingual Equivalence
- Factoid Fast-Path (single-value answers)
- Numeric / Format Equivalence
- Definitive Answer Required
- User name omission allowed

### v1 (core-information match)
Shifted evaluation criterion from *"all information"* to *"core information"*
and added an acceptable-variation catalogue (12 bullets). Retained all
v0-specific rules (cross-lingual, factoid fast-path, definitive answer, numeric
equivalence).

### `omnigaia` (multilingual-aware, all checkers)
Rebased on latest gaia2-core (16+ acceptable variations, greeting/sign-off
handling, currency equivalence, etc.). Expanded scope from USER_MESSAGE_CHECKER
only to 6 checkers. Each override applies the minimal edit needed to make the
gaia2-core prompt work correctly for non-English evaluation.
