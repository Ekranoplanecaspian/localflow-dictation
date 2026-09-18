"""Placeholders in snippets, and the snippet substitution itself."""

from __future__ import annotations

from datetime import datetime

from localflow.cleanup.pipeline import CleanupPipeline
from localflow.cleanup.placeholders import expand
from localflow.config import PostProcessConfig

WHEN = datetime(2026, 3, 9, 14, 5)


def test_the_named_placeholders():
    assert expand("Today is {date}.", WHEN) == "Today is 09 March 2026."
    assert expand("at {time}", WHEN) == "at 14:05"
    assert expand("{day}", WHEN) == "Monday"
    assert expand("{month} {year}", WHEN) == "March 2026"
    assert expand("{now}", WHEN) == "09 March 2026 at 14:05"


def test_a_placeholder_may_carry_its_own_format():
    assert expand("{date:%Y-%m-%d}", WHEN) == "2026-03-09"
    assert expand("{time:%I%p}", WHEN).lower() == "02pm"


def test_names_are_matched_whatever_case_they_are_written_in():
    assert expand("{DATE}", WHEN) == "09 March 2026"


def test_an_unknown_placeholder_is_left_exactly_as_written():
    """`{foo}` is much more likely to be part of the user's own template - a bit of JSON, a
    format string - than a misspelling of one of ours."""
    assert expand("{foo} and {bar:baz}", WHEN) == "{foo} and {bar:baz}"
    assert expand('{"key": "value"}', WHEN) == '{"key": "value"}'


def test_a_broken_format_shows_itself_rather_than_losing_the_dictation():
    out = expand("{date:%Q}", WHEN)
    assert out in ("{date:%Q}", "%Q"), "either untouched or strftime passed it through"


def test_text_with_no_placeholders_is_returned_untouched():
    assert expand("Best,\nArnab") == "Best,\nArnab"


# --- the substitution itself --------------------------------------------------------------------
def _pipeline(snippets):
    cfg = PostProcessConfig()
    cfg.snippets = snippets
    return CleanupPipeline(cfg, None)


def test_a_snippet_containing_a_backslash_is_inserted_literally():
    """The replacement used to go straight into `re.sub`, where "\\1" is a group reference and a
    trailing backslash raises - either way the user's dictation is damaged by their own snippet."""
    out = _pipeline({"code marker": r"see \1 and C:\temp"}).rules("code marker")
    assert r"\1" in out and r"C:\temp" in out


def test_a_snippet_expands_its_placeholders_when_it_fires():
    out = _pipeline({"log header": "Entry for {year}"}).rules("log header")
    assert out == f"Entry for {datetime.now():%Y}"


def test_a_snippet_only_fires_on_a_whole_word():
    out = _pipeline({"sig": "SIGNATURE"}).rules("the design is fine")
    assert "SIGNATURE" not in out
