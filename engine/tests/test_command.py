"""Command mode replaces text the user has already written, so almost every test here is
about refusing to do that."""

from __future__ import annotations

import pytest

from localflow.cleanup.command import CommandRunner, _strip_fences, guard
from localflow.llm.providers import Completion

SELECTION = "we should probably ship this on tuesday if the tests pass"


class FakeProvider:
    """Returns whatever the test wants, and records what it was asked."""

    def __init__(self, text: str, truncated: bool = False, raises: Exception | None = None):
        self.text, self.truncated, self.raises = text, truncated, raises
        self.system: str | None = None
        self.user: str | None = None
        self.max_tokens: int | None = None

    def complete(self, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.0):
        if self.raises:
            raise self.raises
        self.system, self.user, self.max_tokens = system, user, max_tokens
        return Completion(text=self.text, ms=1.0, truncated=self.truncated)

    def prefill(self, system: str, user: str) -> None:  # pragma: no cover - not used here
        pass


def run(model_says: str, selection: str = SELECTION, instruction: str = "make it more formal", **kw):
    return CommandRunner(FakeProvider(model_says, **kw)).run(selection, instruction)


# --- the happy path ---------------------------------------------------------------------------
def test_a_clean_edit_replaces_the_selection():
    r = run("We should ship this on Tuesday if the tests pass.")
    assert r.changed is True
    assert r.text == "We should ship this on Tuesday if the tests pass."
    assert r.rejected is None


def test_an_instruction_may_throw_away_every_original_word():
    """The clean-up guard forbids dropping content words. A command must be allowed to: this is
    what "summarise" and "translate" do, and rejecting it would defeat the whole feature."""
    r = run("Nous devrions livrer mardi.", instruction="translate this to french")
    assert r.changed is True and r.text == "Nous devrions livrer mardi."


def test_a_summary_may_be_far_shorter_than_the_selection():
    long = " ".join(["the meeting covered several topics at some length"] * 8)
    r = run("The meeting covered several topics.", selection=long, instruction="summarise this")
    assert r.changed is True


# --- refusing to touch the user's text --------------------------------------------------------
@pytest.mark.parametrize(
    "model_says, reason",
    [
        ("", "empty"),
        ("   \n  ", "empty"),
        ("I'm sorry, I can't help with that.", "assistant-speak"),
        ("Sure! Here's a more formal version: We should ship on Tuesday.", "assistant-speak"),
        ("Here is the edited text: We should ship on Tuesday.", "assistant-speak"),
        ('"We should ship this on Tuesday."', "wrapped"),
        ("We should ship on Tuesday.\n\nNote: I made it more formal.", "commentary"),
        ("We should ship on Tuesday.\nLet me know if you want another tone.", "commentary"),
    ],
)
def test_a_model_that_breaks_role_never_reaches_the_document(model_says, reason):
    r = run(model_says)
    assert r.changed is False, "the selection must be left alone"
    assert r.text == SELECTION
    assert r.rejected == reason


def test_truncated_output_is_refused_rather_than_pasted_half_written():
    r = run("We should ship this on Tues", truncated=True)
    assert r.changed is False and r.rejected == "truncated" and r.text == SELECTION


def test_a_provider_that_raises_leaves_the_text_alone():
    r = run("ignored", raises=RuntimeError("connection refused"))
    assert r.changed is False and r.text == SELECTION
    assert r.rejected is not None and "connection refused" in r.rejected


def test_no_model_means_no_command():
    r = CommandRunner(None).run(SELECTION, "make it formal")
    assert r.changed is False and r.rejected == "no-model"


@pytest.mark.parametrize(
    "selection, instruction, reason",
    [("", "make it formal", "empty-selection"), (SELECTION, "   ", "empty-instruction")],
)
def test_nothing_to_do_is_not_an_edit(selection, instruction, reason):
    r = CommandRunner(FakeProvider("anything")).run(selection, instruction)
    assert r.changed is False and r.rejected == reason


def test_returning_the_selection_unchanged_is_not_a_replacement():
    """The prompt tells the model to return the text as-is when it cannot apply the
    instruction. Pasting it back over itself would still disturb the caret and the undo stack."""
    r = run(SELECTION)
    assert r.changed is False and r.rejected == "no-change"


def test_echoing_the_instruction_is_treated_as_conversation():
    r = run("make it more formal: We should ship on Tuesday.")
    assert r.changed is False and r.rejected == "echoed-instruction"


# --- things that look like role-breaking but are not ------------------------------------------
def test_an_apology_inside_the_text_being_edited_is_fine():
    """`_REFUSAL` anchors at the start for this reason: the user may well be editing a sentence
    that begins with an apology only *after* something else."""
    r = run("Tell them I am sorry for the delay and that we will ship on Tuesday.",
            selection="tell them sorry about the delay, we ship tuesday",
            instruction="make it a full sentence")
    assert r.changed is True


def test_a_quoted_selection_may_come_back_quoted():
    r = run('"We should ship on Tuesday."', selection='"we ship tuesday"',
            instruction="fix the grammar")
    assert r.changed is True, "the quotes belong to the user's text, not to the model"


# --- shaping the model's output ---------------------------------------------------------------
def test_a_code_fence_around_the_whole_answer_is_removed():
    assert _strip_fences("```\nprint(1)\n```") == "print(1)"
    assert _strip_fences("```python\nprint(1)\n```") == "print(1)"


def test_a_selection_that_is_itself_fenced_code_keeps_its_fences():
    """Editing a fenced block should return a fenced block. Stripping here would quietly eat
    the user's markdown."""
    text = "```python\nprint(1)\n```\n\nAnd some prose.\n\n```js\nconsole.log(1)\n```"
    assert _strip_fences(text) == text


def test_markdown_line_breaks_and_em_dashes_are_tidied_away():
    """Both are invisible in a chat window and litter in a text field. The trailing two spaces
    are what the model actually returned when asked for bullet points."""
    from localflow.cleanup.command import _tidy

    assert _tidy("- fix the bug  \n- update the docs  \n") == "- fix the bug\n- update the docs"
    assert _tidy("ship it — if the tests pass") == "ship it - if the tests pass"
    assert _tidy("ship it—maybe") == "ship it - maybe"


def test_the_guard_says_nothing_about_length():
    assert guard("a short line", "a" * 4000, "expand this") is None
    assert guard(" ".join(["word"] * 200), "Tiny.", "summarise") is None


def test_generation_budget_leaves_room_to_expand_but_is_not_unbounded():
    short, long = CommandRunner(None), CommandRunner(None)
    assert short.budget("hello there") >= 96
    assert long.budget(" ".join(["word"] * 100)) > short.budget("hello there")
