"""The clean-up settings, checked before they are used.

Everything the Hub sends - and whatever is in a hand-edited config file - passes through here.
Values are tidied where the intent is clear (spaces around a word, the newline a pasted API key
brings with it, a term listed twice), and refused where it is not: a refused value keeps what was
there before, and the reason goes back to the Hub. Some of these were worse than untidy: a snippet
with an empty trigger matched between every two words and was inserted all through every
dictation, and custom instructions longer than the clean-up model's context made every clean-up
fail.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any
from urllib.parse import urlsplit

from localflow.config import PostProcessConfig

PROVIDERS = ("bundled", "ollama", "openai", "anthropic")
MAX_TERMS = 2000
MAX_TERM_CHARS = 80
MAX_REPLACEMENTS = 2000
MAX_SNIPPETS = 500
MAX_TRIGGER_CHARS = 60
MAX_SNIPPET_CHARS = 5000
MAX_INSTRUCTION_CHARS = 1500  # the bundled model has a 2048-token context for everything


def _text_map(value: Any, what: str, key_max: int, value_max: int, limit: int,
              problems: list[str]) -> dict[str, str] | None:
    if not isinstance(value, dict):
        problems.append(f"{what} must be a list of pairs")
        return None
    out: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not isinstance(v, str):
            problems.append(f"{what}: {k!r} is not text")
            continue
        k, v = k.strip(), v.strip()
        if not k:
            continue  # an empty key matches everywhere: never kept
        if len(k) > key_max or len(v) > value_max:
            problems.append(f"{what}: {k[:30]!r} is too long")
            continue
        out[k] = v
    if len(out) > limit:
        problems.append(f"{what}: only the first {limit} are kept")
        out = dict(list(out.items())[:limit])
    return out


def _check_url(url: Any) -> str | None:
    """The address, tidied, or None when it is not one."""
    if not isinstance(url, str):
        return None
    url = url.strip().rstrip("/")
    if not url:
        return ""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return url


def check_postprocess(new: PostProcessConfig, old: PostProcessConfig) -> tuple[PostProcessConfig, list[str]]:
    """`new`, tidied, with anything unusable put back to `old`; and a plain reason for each."""
    problems: list[str] = []
    fix: dict[str, Any] = {}

    for f in fields(PostProcessConfig):
        value, before = getattr(new, f.name), getattr(old, f.name)
        if isinstance(before, bool) and not isinstance(value, bool):
            problems.append(f"{f.name} must be on or off")
            fix[f.name] = before

    terms = new.dictionary_terms
    if not isinstance(terms, list):
        problems.append("dictionary words must be a list")
        fix["dictionary_terms"] = old.dictionary_terms
    else:
        seen, kept = set(), []
        for t in terms:
            if not isinstance(t, str):
                continue
            t = " ".join(t.split())
            if not t or t.lower() in seen:
                continue
            if len(t) > MAX_TERM_CHARS:
                problems.append(f"dictionary word {t[:30]!r}... is too long")
                continue
            seen.add(t.lower())
            kept.append(t)
        if len(kept) > MAX_TERMS:
            problems.append(f"only the first {MAX_TERMS} dictionary words are kept")
            kept = kept[:MAX_TERMS]
        fix["dictionary_terms"] = kept

    for name, what, kmax, vmax, limit in (
        ("dictionary", "replacements", MAX_TERM_CHARS, MAX_TERM_CHARS, MAX_REPLACEMENTS),
        ("snippets", "snippets", MAX_TRIGGER_CHARS, MAX_SNIPPET_CHARS, MAX_SNIPPETS),
    ):
        tidy = _text_map(getattr(new, name), what, kmax, vmax, limit, problems)
        fix[name] = tidy if tidy is not None else getattr(old, name)

    instructions = new.custom_instructions if isinstance(new.custom_instructions, str) else old.custom_instructions
    instructions = instructions.strip()
    if len(instructions) > MAX_INSTRUCTION_CHARS:
        problems.append(f"your own instructions can be up to {MAX_INSTRUCTION_CHARS} characters; "
                        "the clean-up model has room for little more")
        instructions = old.custom_instructions
    fix["custom_instructions"] = instructions

    if new.llm_provider not in PROVIDERS:
        problems.append(f"{new.llm_provider!r} is not a clean-up provider")
        fix["llm_provider"] = old.llm_provider
    url = _check_url(new.llm_url)
    if url is None:
        problems.append(f"{new.llm_url!r} is not a web address; it needs to start with http:// or https://")
        fix["llm_url"] = old.llm_url
    else:
        fix["llm_url"] = url
    key = new.llm_api_key if isinstance(new.llm_api_key, str) else old.llm_api_key
    fix["llm_api_key"] = "".join(key.split())  # a pasted key often brings a newline with it
    fix["llm_model"] = new.llm_model.strip() if isinstance(new.llm_model, str) else old.llm_model

    for name, low, high in (("llm_min_words", 0, 100), ("llm_max_tokens", 16, 4000)):
        value = getattr(new, name)
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            problems.append(f"{name} must be a whole number from {low} to {high}")
            fix[name] = getattr(old, name)
    timeout = new.llm_timeout_s
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 1 <= timeout <= 120:
        problems.append("the clean-up timeout must be between 1 and 120 seconds")
        fix["llm_timeout_s"] = old.llm_timeout_s

    return replace(new, **fix), problems


def check_mirror(url: Any) -> tuple[str | None, str | None]:
    """(the mirror address tidied, or None when refused; the reason). "" means huggingface.co."""
    tidy = _check_url(url)
    if tidy is None:
        return None, "the download mirror must be an http:// or https:// address"
    return tidy, None
