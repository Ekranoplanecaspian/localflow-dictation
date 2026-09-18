"""Personal dictionary with phonetic matching.

Speech models spell unfamiliar names the way they sound: "Arnub", "Okonko", "parasit".
Given the right spellings, we match transcript words that sound like a dictionary term
(same Metaphone code, or a high Jaro-Winkler similarity) and replace them, including
two-word spans for names like "Dr Okonkwo" or "Wispr Flow". Common English words are never
touched, so "sit" does not become "Smith".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import jellyfish

log = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
JW_THRESHOLD = 0.86
MIN_LEN = 3
# spoken forms of titles that dictionary terms usually abbreviate
_ALIASES = {"doctor": "dr", "mister": "mr", "missus": "mrs", "miss": "ms", "professor": "prof", "saint": "st"}

# Words that are never replaced even if they sound like a term. Kept short: a term that is a
# common word ("Flow", "Mail") is matched only as part of a multi-word term.
_COMMON = set("""
the a an and or but if then than that this these those there their they them we you your our
i me my he she it its his her him is are was were be been being have has had do does did done
not no yes so as at by for from in into of on to with without about after before over under
up down out off again more most some any all each other such only own same very can will just
should would could may might must shall get got make made go went come came see saw say said
know knew think thought take took give gave find found tell told ask asked work call try need
feel leave put keep let begin seem help talk turn start show hear play run move live believe
hold bring happen write provide sit stand lose pay meet include continue set learn change lead
understand watch follow stop create speak read allow add spend grow open walk win offer remember
love consider appear buy wait serve die send expect build stay fall cut reach kill remain
one two three four five six seven eight nine ten first second third next last new old good bad
big small long short high low great little right left early late young day week month year time
way thing man woman people person life hand part place case point number group problem fact
""".split())


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    original: str
    replacement: str
    score: float


def _sig(word: str) -> str:
    return jellyfish.metaphone(word) or ""


class Dictionary:
    def __init__(self, terms: list[str], replacements: dict[str, str] | None = None):
        # explicit replacements ("heard" -> "meant") are applied first, exactly, case-insensitively
        self.replacements = [(re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v) for k, v in (replacements or {}).items()]
        self.terms: list[tuple[str, list[str], list[str]]] = []  # (term, words, metaphones)
        for term in terms:
            words = term.split()
            if not words:
                continue
            self.terms.append((term, [w.lower() for w in words], [_sig(w) for w in words]))

    # ----------------------------------------------------------------------------------
    def apply(self, text: str) -> tuple[str, list[Match]]:
        for pat, rep in self.replacements:
            text = pat.sub(rep, text)
        if not self.terms:
            return text, []
        tokens = list(_WORD.finditer(text))
        matches: list[Match] = []
        i = 0
        while i < len(tokens):
            best: Match | None = None
            for term, words, sigs in self.terms:
                n = len(words)
                if i + n > len(tokens):
                    continue
                span = tokens[i:i + n]
                cand = [t.group(0) for t in span]
                score = self._score(cand, words, sigs)
                if score is not None and (best is None or score > best.score):
                    best = Match(span[0].start(), span[-1].end(), " ".join(cand), term, score)
            if best is not None and best.original.lower() != best.replacement.lower():
                matches.append(best)
                i += len(best.replacement.split())
            else:
                i += 1
        if not matches:
            return text, []
        out, pos = [], 0
        for m in matches:
            out.append(text[pos:m.start])
            out.append(m.replacement)
            pos = m.end
        out.append(text[pos:])
        return "".join(out), matches

    def _score(self, cand: list[str], words: list[str], sigs: list[str]) -> float | None:
        total = 0.0
        for c, w, s in zip(cand, words, sigs):
            cl = c.lower()
            cl = _ALIASES.get(cl, cl)
            w_cmp = w.rstrip(".")
            if cl == w_cmp:
                total += 1.0
                continue
            if len(cl) < MIN_LEN or (cl in _COMMON and len(words) == 1):
                return None
            jw = jellyfish.jaro_winkler_similarity(cl, w_cmp)
            same_sound = bool(s) and _sig(c) == s
            edits = jellyfish.damerau_levenshtein_distance(cl, w_cmp)
            close_edit = edits <= (1 if len(w_cmp) >= 4 else 0) or (edits <= 2 and len(w_cmp) >= 8)
            if same_sound and jw >= 0.7:
                total += max(jw, 0.9)
            elif jw >= JW_THRESHOLD or (close_edit and jw >= 0.75):
                total += max(jw, 0.85)
            else:
                return None
        return total / len(words)
