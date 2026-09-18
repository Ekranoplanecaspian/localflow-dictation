"""Style profiles: what "clean" means depends on where the text is going."""

from __future__ import annotations

from dataclasses import dataclass

CHAT_APPS = {"slack", "discord", "teams", "ms-teams", "whatsapp", "telegram", "signal", "messenger", "olk"}
EMAIL_APPS = {"outlook", "thunderbird", "hxoutlook", "mailspring"}
DOCS_APPS = {"winword", "notion", "obsidian", "wordpad", "onenote", "typora", "evernote", "logseq"}
CODE_APPS = {"code", "code - insiders", "cursor", "windsurf", "devenv", "pycharm64", "idea64", "clion64", "webstorm64",
             "rider64", "sublime_text", "notepad++", "zed", "claude"}
TERMINAL_APPS = {"windowsterminal", "wt", "cmd", "powershell", "pwsh", "alacritty", "mintty", "conhost", "wezterm-gui"}
BROWSERS = {"chrome", "msedge", "brave", "firefox", "arc", "opera", "vivaldi", "zen"}


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    instructions: str


PROFILES = {
    "default": Profile("default", "General", "Write clean, natural prose with normal punctuation and capitalisation."),
    "chat": Profile("chat", "Chat", "This is a chat message. Casual tone; sentence fragments are fine; no formal "
                    "salutations or sign-offs added; keep the speaker's words and tone exactly (do not make it terser)."),
    "email": Profile("email", "Email", "This is an email. Use complete sentences and paragraphs; keep greetings and "
                     "sign-offs the speaker said; formal but natural."),
    "docs": Profile("docs", "Document", "This is a document. Use complete sentences and paragraphs; format dictated "
                    "lists as lists; keep terminology exact."),
    "code": Profile("code", "Code editor", "This is going into a code editor or an AI coding assistant. Preserve "
                    "identifiers, file names, commands and symbols exactly as spoken (camelCase, snake_case, paths, "
                    "flags); do not add punctuation inside code; keep technical terms verbatim."),
    "terminal": Profile("terminal", "Terminal", "This is a terminal. Output the command or text exactly; no trailing "
                        "period; no capitalisation changes; no commentary."),
}


def profile_for(app: str | None, title: str | None = None, override: str | None = None) -> Profile:
    """Which style profile applies here.

    `override` is a per-app rule the user set in the Hub and beats everything below it: the
    guesses here are good for the apps they know about and useless for a bespoke internal tool,
    and being able to say "this one is email" is the whole point of a per-app rule.
    """
    if override and override != "auto":
        chosen = PROFILES.get(override)
        if chosen is not None:
            return chosen
    name = (app or "").lower().removesuffix(".exe")
    t = (title or "").lower()
    if name in CHAT_APPS:
        return PROFILES["chat"]
    if name in EMAIL_APPS:
        return PROFILES["email"]
    if name in DOCS_APPS:
        return PROFILES["docs"]
    if name in CODE_APPS:
        return PROFILES["code"]
    if name in TERMINAL_APPS:
        return PROFILES["terminal"]
    if name in BROWSERS:
        if any(k in t for k in ("gmail", "outlook", "mail", "proton")):
            return PROFILES["email"]
        if any(k in t for k in ("slack", "discord", "whatsapp", "teams", "messenger", "telegram")):
            return PROFILES["chat"]
        if any(k in t for k in ("google docs", "notion", "confluence", "- word", "document")):
            return PROFILES["docs"]
        if any(k in t for k in ("github", "gitlab", "stack overflow", "chatgpt", "claude", "colab", "jupyter")):
            return PROFILES["code"]
    return PROFILES["default"]
