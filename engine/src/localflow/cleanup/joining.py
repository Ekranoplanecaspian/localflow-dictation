"""Fit dictated text to the place it is about to land.

Clean-up treats every take as a complete utterance: it capitalises the first word and ends with
a full stop, which is right when the caret is at the start of an empty field and wrong the rest
of the time. Dictating "and then we can ship it" into the middle of a half-typed sentence
produced "...I think And then we can ship it." - a capital in the middle of a clause and no
space before it.

The shell already reads the text immediately before the caret through UI Automation and sends
it with every session; until now the engine threw it away. This module is what reads it.

Two decisions, both deliberately conservative, because the cost of each mistake is asymmetric:

* **The space.** Cheap to get wrong in either direction and easy to see, so the rules here are
  mechanical.
* **The capital.** Lower-casing a proper noun ("Priya" -> "priya") is a real corruption of the
  user's words, while leaving an unwanted capital is a blemish they can see and fix. So a
  leading capital is only ever removed when the first word is a closed-class function word -
  the ones that open a continuation ("and", "but", "the", "so", "which") and are essentially
  never names. Anything else keeps whatever case it arrived with.
"""

from __future__ import annotations

import re

#: Characters that end a sentence. A colon or semicolon does not: what follows them continues
#: the same sentence and stays lower-case.
SENTENCE_END = ".!?…"
#: Closing marks that may sit after the full stop: `He left.")` still ends a sentence.
CLOSERS = ")]}\"'”’"
#: After these, dictated text joins straight on with no space - an opening bracket or quote, or
#: something that is plainly mid-token like a hyphen, slash or @.
NO_SPACE_AFTER = "([{<\"'“‘/\\-–—@#$_~"

#: Words that may lose a sentence-opening capital. Closed classes only: determiners, pronouns,
#: conjunctions, prepositions, auxiliaries and the handful of adverbs people actually start a
#: continuation with. No nouns, and nothing that doubles as a name.
FUNCTION_WORDS = frozenset(
    """
a an the this that these those
and but or nor so yet for because since although though while whereas unless until if when
whenever wherever where which who whom whose what why how than as
i you he she it we they me him her us them my your his its our their mine yours ours theirs
myself yourself himself herself itself ourselves themselves
is are was were be been being am do does did doing have has had having
can could shall should will would may might must ought need dare
in on at by to from with without within into onto upon over under above below between among
across through during before after about against along around behind beside besides beyond
near off out past per plus toward towards via
not no nor never always often sometimes usually rarely again also too very quite rather just
only even still yet already almost nearly perhaps maybe probably possibly actually basically
essentially obviously clearly simply really truly literally honestly frankly
here there then now today tonight thus hence therefore however moreover furthermore meanwhile
otherwise instead anyway besides regardless
all any both each either every few many most much neither none one other others several some
such
""".split()
)

_FIRST_WORD = re.compile(r"^([^\W\d_]+)", re.UNICODE)


def _last_meaningful(before: str) -> str:
    """The last character that decides sentence position, ignoring closing quotes and brackets."""
    s = before.rstrip()
    i = len(s) - 1
    while i >= 0 and s[i] in CLOSERS:
        i -= 1
    return s[i] if i >= 0 else ""


def continues_sentence(before: str) -> bool:
    """Whether the caret sits inside a sentence that is already under way."""
    if not before or not before.strip():
        return False
    # A line break starts something new even with no punctuation: a new list item, a new
    # paragraph, the next cell.
    if before.rstrip(" \t").endswith(("\n", "\r")):
        return False
    last = _last_meaningful(before)
    if not last:
        return False
    return last not in SENTENCE_END


def needs_space(before: str) -> bool:
    """Whether a space belongs between what is there and what is about to be inserted."""
    if not before:
        return False
    last = before[-1]
    if last.isspace():
        return False
    return last not in NO_SPACE_AFTER


def lower_first(text: str) -> str:
    """Drop a sentence-opening capital, but only from a word that could not be a name."""
    match = _FIRST_WORD.match(text)
    if not match:
        return text
    word = match.group(1)
    if not word[:1].isupper():
        return text
    # "I" and "I'm" keep their capital wherever they appear.
    if word == "I":
        return text
    # An acronym or CamelCase token is not a sentence-opening capital: "NASA", "iPhone",
    # "GitHub" all carry their own case and must survive untouched.
    if any(c.isupper() for c in word[1:]):
        return text
    if word.lower() not in FUNCTION_WORDS:
        return text
    return word[0].lower() + text[len(word[0]) :]


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
#: How far back to look for an echo. Long enough to catch a repeated clause, short enough that
#: two genuinely similar sentences in a row cannot collide by accident.
ECHO_WINDOW = 12
#: One repeated word is a coincidence ("so" after "and so"); two in a row is the model copying.
ECHO_MIN = 2


def strip_echo(before: str, text: str) -> str:
    """Remove the tail of `before` when the model has repeated it at the start of `text`.

    Showing the model the text at the caret helps it match names and casing, but a model handed
    a half-finished sentence tends to finish it - including the part it was only meant to read.
    Measured with Qwen3-4B: before "I spoke to", dictating "priya about the parakeet rollout"
    came back as "I spoke to Priya about the parakeet rollout", which would have been inserted
    after the existing "I spoke to" and read "I spoke to I spoke to Priya".

    The prompt already forbids this and the model does it anyway, so it is undone here instead
    of argued about there.
    """
    if not before.strip() or not text.strip():
        return text
    tail = [m.group(0).lower() for m in _WORD.finditer(before)][-ECHO_WINDOW:]
    if len(tail) < ECHO_MIN:
        return text
    starts = [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]
    # Longest first: prefer removing the whole repeated clause over part of it.
    for n in range(min(len(tail), len(starts)), ECHO_MIN - 1, -1):
        if [w for w, _, _ in starts[:n]] == tail[-n:]:
            # Whatever punctuation closed the echoed clause belongs to the echo, not to the
            # new text: leaving it behind inserts ". We shipped." after the existing sentence.
            return text[starts[n - 1][2] :].lstrip(" \t.,;:!?-–—")
    return text


def join(before: str | None, text: str) -> str:
    """Adjust `text` so it reads correctly inserted at a caret preceded by `before`.

    Returns the text with its leading space and its first letter's case settled. Everything
    else - punctuation, the words themselves - is left exactly as clean-up produced it.
    """
    if not text:
        return text
    before = before or ""
    if not before.strip():
        return text

    out = strip_echo(before, text)
    if continues_sentence(before):
        out = lower_first(out)
    if needs_space(before):
        out = " " + out
    return out
