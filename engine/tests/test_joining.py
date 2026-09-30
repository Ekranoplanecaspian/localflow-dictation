"""Fitting dictated text to the caret it lands at.

The capitalisation tests matter more than the spacing ones: a missing space is visible and
trivial to fix, while lower-casing somebody's name is a corruption of what they said.
"""

from __future__ import annotations

import pytest

from localflow.cleanup.joining import continues_sentence, join, lower_first, needs_space


# --- the case that started this ----------------------------------------------------------------
def test_dictating_into_the_middle_of_a_sentence():
    """The reported bug: "...I think" + "And then we can ship it." came out with a capital A
    and no space in front of it."""
    assert join("I think", "And then we can ship it.") == " and then we can ship it."


def test_dictating_after_a_finished_sentence_keeps_its_capital():
    assert join("We shipped it.", "The tests all passed.") == " The tests all passed."


def test_an_empty_field_is_left_completely_alone():
    assert join("", "Hello there.") == "Hello there."
    assert join(None, "Hello there.") == "Hello there."
    assert join("   \n  ", "Hello there.") == "Hello there."


# --- spacing ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "before, expected",
    [
        ("I think", True),
        ("I think ", False),
        ("I think\n", False),
        ("", False),
        ("see (", False),
        ('he said "', False),
        ("well-", False),
        ("path/", False),
        ("mail@", False),
    ],
)
def test_when_a_space_is_wanted(before, expected):
    assert needs_space(before) is expected


def test_no_space_is_added_after_an_opening_bracket():
    assert join("see (", "the appendix") == "the appendix"


# --- sentence position ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "before, expected",
    [
        ("I think", True),
        ("we should ship it,", True),
        ("first:", True, ),
        ("one thing;", True),
        ("We shipped it.", False),
        ("Did we ship it?", False),
        ("Ship it!", False),
        ('He said "we shipped it."', False),
        ("We shipped it.)", False),
        ("a new line\n", False),
        ("", False),
    ],
)
def test_where_the_caret_sits(before, expected):
    assert continues_sentence(before) is expected


def test_a_colon_continues_the_sentence_but_a_full_stop_does_not():
    """After a colon what follows is still the same sentence, so it stays lower-case."""
    assert join("The problem is this:", "It never shipped.") == " it never shipped."
    assert join("The problem is this.", "It never shipped.") == " It never shipped."


# --- capitalisation: what must never be touched --------------------------------------------------
def test_a_name_keeps_its_capital_mid_sentence():
    """The whole reason `lower_first` works off a closed word list. Getting this wrong rewrites
    the user's words rather than merely leaving a blemish."""
    assert join("I spoke to", "Priya about the release.") == " Priya about the release."
    assert join("we use", "Parakeet for speech.") == " Parakeet for speech."


@pytest.mark.parametrize(
    "word", ["NASA", "GitHub", "iPhone", "LocalFlow", "McDonald", "PyTorch"]
)
def test_acronyms_and_camel_case_survive(word):
    assert lower_first(f"{word} is fine.") == f"{word} is fine."


def test_the_pronoun_i_keeps_its_capital():
    assert join("she said", "I will ship it.") == " I will ship it."
    assert join("she said", "I'm shipping it.") == " I'm shipping it."


def test_days_and_months_are_not_function_words():
    assert join("ship it on", "Tuesday if the tests pass.") == " Tuesday if the tests pass."
    assert join("due in", "March at the latest.") == " March at the latest."


def test_an_unknown_word_keeps_whatever_case_it_arrived_with():
    """Conservative by design: if it is not on the list, it is left alone rather than guessed at."""
    assert lower_first("Kubernetes rolled it back.") == "Kubernetes rolled it back."


# --- capitalisation: what should be lowered ------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("And then we shipped.", "and then we shipped."),
        ("But it failed.", "but it failed."),
        ("The tests passed.", "the tests passed."),
        ("So we waited.", "so we waited."),
        ("Which is why it broke.", "which is why it broke."),
        ("Because the tests failed.", "because the tests failed."),
        ("Maybe on Tuesday.", "maybe on Tuesday."),
        ("Actually, never mind.", "actually, never mind."),
    ],
)
def test_a_continuation_opener_loses_its_capital(text, expected):
    assert lower_first(text) == expected


def test_text_that_is_already_lower_case_is_untouched():
    assert lower_first("and then we shipped.") == "and then we shipped."


def test_leading_punctuation_does_not_confuse_the_first_word():
    assert lower_first('"And then we shipped."') == '"And then we shipped."'


# --- the preceding text also reaches the clean-up model ------------------------------------------
def test_the_prompt_carries_the_text_before_the_caret_as_context_only():
    """Wiring a value through and then never using it is an easy mistake to make and an
    invisible one, so this checks the prompt itself."""
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig
    from localflow.cleanup.profiles import profile_for

    pipeline = CleanupPipeline(PostProcessConfig(), None)
    prompt = pipeline.system_prompt(profile_for(None, None), "Priya asked about the Parakeet rollout")

    assert "Priya asked about the Parakeet rollout" in prompt
    # It must be framed as reference material, not as something to carry on writing.
    assert "Do not continue it" in prompt


def test_a_very_long_preceding_passage_is_trimmed():
    """A whole document before the caret would swamp the prompt and slow every dictation."""
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig
    from localflow.cleanup.profiles import profile_for

    pipeline = CleanupPipeline(PostProcessConfig(), None)
    plain = pipeline.system_prompt(profile_for(None, None))
    prompt = pipeline.system_prompt(profile_for(None, None), "x" * 5000)
    # Only the last 400 characters survive, whatever was in front of them.
    assert "x" * 400 in prompt
    assert "x" * 401 not in prompt
    assert len(prompt) - len(plain) < 700, "the context costs a fixed, small amount"


def test_no_preceding_text_adds_nothing_to_the_prompt():
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig
    from localflow.cleanup.profiles import profile_for

    pipeline = CleanupPipeline(PostProcessConfig(), None)
    plain = pipeline.system_prompt(profile_for(None, None))
    assert "immediately before the caret" not in plain


# --- the model repeating the text it was only shown for reference --------------------------------
def test_the_measured_echo_is_undone():
    """Qwen3-4B, given "I spoke to" as context and "priya about the parakeet rollout" as the
    take, returned the context back at the front of its answer. Inserted at the caret that
    reads "I spoke to I spoke to Priya"."""
    from localflow.cleanup.joining import strip_echo

    assert (
        strip_echo("I spoke to", "I spoke to Priya about the parakeet rollout.")
        == "Priya about the parakeet rollout."
    )
    assert join("I spoke to", "I spoke to Priya about the rollout.", from_model=True) == " Priya about the rollout."


def test_words_the_speaker_repeated_are_kept_when_no_model_wrote_them():
    """Only the model echoes. Without it, repeating what is already at the caret is what the
    user said, and it used to be cut: "Thanks so much for the help" came out "for the help"."""
    assert join("Thanks so much.", "Thanks so much for the help.") == " Thanks so much for the help."
    assert join("we should ship it", "We should ship it") == " we should ship it"


def test_the_engine_only_strips_an_echo_from_model_output(monkeypatch):
    """The whole path: the same take, cleaned with rules only and then by a model that echoes."""
    import time

    import numpy as np

    import localflow.service.engine as eng
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import Config, PostProcessConfig
    from tests.test_cleanup import FakeProvider
    from tests.test_service import FakeSTT, blocks, fixture_audio

    class SaysThanks(FakeSTT):
        def transcribe(self, audio, language=None):
            return "Thanks so much for the help."

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: SaysThanks())
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    engine = eng.Engine(cfg)
    engine.load()
    deadline = time.monotonic() + 30
    while engine.state == "loading" and time.monotonic() < deadline:
        time.sleep(0.05)

    def final_text(pipeline: CleanupPipeline) -> str:
        engine.cleanup = pipeline
        events: list[dict] = []
        s = engine.start_session("s", {"before_caret": "Thanks so much."}, events.append)
        for b in blocks(fixture_audio()):
            s.feed((np.clip(b * 32767, -32768, 32767)).astype("<i2").tobytes())
        s.end()
        deadline = time.monotonic() + 30
        while not s.done and time.monotonic() < deadline:
            time.sleep(0.05)
        return next(e["text"] for e in events if e["type"] == "final")

    try:
        rules_only = final_text(CleanupPipeline(PostProcessConfig(llm_cleanup=False)))
        assert rules_only == " Thanks so much for the help."
        echoing = FakeProvider("Thanks so much. Thanks so much for the help.")
        with_model = final_text(CleanupPipeline(PostProcessConfig(llm_cleanup=True, llm_min_words=1), echoing))
        assert with_model == " Thanks so much for the help."
    finally:
        engine.shutdown()


def test_an_echo_is_matched_ignoring_case_and_punctuation():
    from localflow.cleanup.joining import strip_echo

    assert strip_echo("the release notes are done,", "The release notes are done. We shipped.") == "We shipped."


def test_a_single_repeated_word_is_left_alone():
    """"and" following "...and" is ordinary English, not a model copying its context."""
    from localflow.cleanup.joining import strip_echo

    assert strip_echo("we tested it and", "and then shipped it.") == "and then shipped it."


def test_text_that_merely_resembles_the_context_is_not_truncated():
    from localflow.cleanup.joining import strip_echo

    assert strip_echo("I spoke to Priya", "She was happy with it.") == "She was happy with it."


def test_an_echo_of_everything_leaves_nothing_to_insert():
    """Degenerate but real: the model returns only the context. Better an empty insert than
    silently duplicating the user's sentence."""
    from localflow.cleanup.joining import strip_echo

    assert strip_echo("we should ship it", "We should ship it").strip() == ""
