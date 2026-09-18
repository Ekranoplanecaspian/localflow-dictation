"""Inverse text normalisation: the unambiguous spoken forms that should become symbols.

Only rules that are practically never wrong live here. Anything contextual (whether "three
thirty" is 3:30 or "three thirty") is left to the language model.
"""

from __future__ import annotations

import re

_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
_ONES["oh"] = 0
_DIGIT_WORD = "|".join(_ONES)
# three or more spoken digits in a row: phone numbers, invoice numbers, codes
_DIGIT_RUN = re.compile(rf"\b((?:(?:{_DIGIT_WORD})[ ,-]+){{2,}}(?:{_DIGIT_WORD}))\b", re.IGNORECASE)
# "name at domain dot com" (domain may itself contain "dot")
_TLD = "com|org|net|io|ai|dev|edu|gov|co|in|uk|de|fr|app|me|info"
# the domain may already be dotted because the URL rule ran first ("company.co.uk")
_EMAIL = re.compile(
    rf"\b([a-z0-9][a-z0-9._\-]*(?:\s+(?:dot|underscore)\s+[a-z0-9]+)*)\s+at\s+"
    rf"([a-z0-9\-]+(?:(?:\s+dot\s+|\.)[a-z0-9\-]+)*)(?:\s+dot\s+|\.)({_TLD})\b",
    re.IGNORECASE,
)
# A URL is only unambiguous with a path ("... slash setup"), a "www" prefix, or at least two
# host labels ("docs dot localflow dot dev"). "the dot com boom" must stay words.
_URL = re.compile(
    rf"\b(?:(www\s+dot\s+[a-z0-9\-]+(?:\s+dot\s+[a-z0-9\-]+)*)|([a-z0-9\-]+(?:\s+dot\s+[a-z0-9\-]+)+))\s+dot\s+({_TLD})"
    rf"(?:\s+slash\s+([a-z0-9\-_/]+(?:\s+slash\s+[a-z0-9\-_]+)*))?\b"
    rf"|\b([a-z0-9\-]+)\s+dot\s+({_TLD})\s+slash\s+([a-z0-9\-_/]+(?:\s+slash\s+[a-z0-9\-_]+)*)\b",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s+percent\b", re.IGNORECASE)
_DEGREES = re.compile(r"\b(\d+)\s+degrees\b", re.IGNORECASE)


def _digits(run: str) -> str:
    return "".join(str(_ONES[w.lower()]) for w in re.split(r"[ ,-]+", run.strip()))


def _local(part: str) -> str:
    return re.sub(r"\s+dot\s+", ".", re.sub(r"\s+underscore\s+", "_", part.strip()), flags=re.IGNORECASE).replace(" ", "")


# words that can precede "at <domain>" without being a mailbox name
_NOT_LOCAL = set("""are is am be been being was were it its we they you he she i me us him her them here there
this that these those up down look looks looking meet meeting stay staying work working arrive arriving
start starts starting back still now then available live lives living based sits sit located""".split())


def _email(m: re.Match) -> str:
    local = m.group(1)
    if local.lower().split()[0] in _NOT_LOCAL:
        return m.group(0)
    return f"{_local(local).lower()}@{_local(m.group(2)).lower()}.{m.group(3).lower()}"


def apply(text: str) -> str:
    text = _DIGIT_RUN.sub(lambda m: _digits(m.group(1)), text)
    text = _URL.sub(lambda m: _url(m), text)  # before emails: "... at docs dot x dot dev slash setup" is a URL
    text = _EMAIL.sub(_email, text)
    text = _PERCENT.sub(r"\1%", text)
    text = _DEGREES.sub(r"\1°", text)
    return text


def _url(m: re.Match) -> str:
    if m.group(5):  # single label + tld + path
        host, tld, path = m.group(5), m.group(6), m.group(7)
    else:
        host, tld, path = (m.group(1) or m.group(2)), m.group(3), m.group(4)
    url = f"{_local(host).lower()}.{tld.lower()}"
    if path:
        url += "/" + re.sub(r"\s+slash\s+", "/", path.strip(), flags=re.IGNORECASE).replace(" ", "")
    return url
