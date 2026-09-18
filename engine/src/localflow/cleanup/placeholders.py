"""Placeholders inside snippets.

A snippet turns a spoken trigger into fixed text: say "my signature" and get the sign-off. The
moment anyone uses one for a note or a log line they want today's date in it, and a snippet that
cannot do that has to be finished by hand every time, which defeats the point of it.

Kept deliberately small. These expand at the moment of dictation and nothing else is evaluated,
so a snippet can never run anything - it is a template, not a script.
"""

from __future__ import annotations

import re
from datetime import datetime

#: `{date}`, `{time:%H:%M}` - a name, optionally followed by a strftime format.
_PLACEHOLDER = re.compile(r"\{(date|time|day|month|year|now)(?::([^{}]{1,40}))?\}", re.IGNORECASE)

#: What each name means with no format given. Written the way a person types them rather than
#: ISO, because these land in prose.
DEFAULTS = {
    "date": "%d %B %Y",
    "time": "%H:%M",
    "day": "%A",
    "month": "%B",
    "year": "%Y",
    "now": "%d %B %Y at %H:%M",
}


def expand(value: str, now: datetime | None = None) -> str:
    """Replace the placeholders in a snippet's text.

    An unknown placeholder is left exactly as written: `{foo}` is far more likely to be a piece
    of the user's own template - a code snippet, a JSON body - than a typo for one of these.
    """
    if "{" not in value:
        return value
    moment = now or datetime.now()

    def one(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        fmt = match.group(2) or DEFAULTS[name]
        try:
            return moment.strftime(fmt)
        except (ValueError, TypeError):
            # A malformed format string is the user's typo, and showing it back to them is more
            # useful than an exception that loses the whole dictation.
            return match.group(0)

    return _PLACEHOLDER.sub(one, value)
