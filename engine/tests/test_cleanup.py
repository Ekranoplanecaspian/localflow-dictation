from localflow.cleanup import itn
from localflow.cleanup.dictionary import Dictionary
from localflow.cleanup.profiles import profile_for


def test_itn_digit_runs():
    assert itn.apply("I think it's four four seven one, not four four one seven") == "I think it's 4471, not 4417"
    assert itn.apply("ends in eight eight two three") == "ends in 8823"
    assert itn.apply("two eggs and one pinch") == "two eggs and one pinch"  # never touches short runs


def test_itn_email_and_url():
    assert itn.apply("my email is arnab at example dot com") == "my email is arnab@example.com"
    assert itn.apply("write to first dot last at company dot co dot uk") == "write to first.last@company.co.uk"
    assert itn.apply("the docs are at docs dot localflow dot dev slash setup") == "the docs are at docs.localflow.dev/setup"
    assert itn.apply("look at the dot com boom") == "look at the dot com boom"  # no host: untouched


def test_itn_percent_and_degrees():
    assert itn.apply("up 18 percent year over year") == "up 18% year over year"
    assert itn.apply("it hit 31 degrees") == "it hit 31°"


def test_dictionary_phonetic_names():
    d = Dictionary(["Arnab", "Priya", "Okonkwo", "Parakeet", "Qwen"])
    out, matches = d.apply("please add Riya, Arnub and Doctor Okonko to the invite; we're using parasit and quen")
    assert out == "please add Priya, Arnab and Doctor Okonkwo to the invite; we're using Parakeet and Qwen"
    assert {m.original for m in matches} == {"Riya", "Arnub", "Okonko", "parasit", "quen"}


def test_dictionary_leaves_common_words_alone():
    d = Dictionary(["Smith", "Flow"])
    out, matches = d.apply("sit down and let the water flow")
    assert out == "sit down and let the water flow" and matches == []


def test_dictionary_multiword_terms_and_explicit_replacements():
    d = Dictionary(["Wispr Flow", "Dr Okonkwo"], replacements={"local flow": "LocalFlow"})
    out, _ = d.apply("whisper flow is what local flow replaces, says doctor okonko")
    assert out == "Wispr Flow is what LocalFlow replaces, says Dr Okonkwo"


def test_dictionary_replacements_go_in_exactly_backslashes_and_all():
    """They were used as re.sub templates: "C:\\temp" gained a tab and "C:\\projects\\notes"
    raised (found by an outside review of 0.2.3)."""
    d = Dictionary([], replacements={
        "temp folder": r"C:\temp",
        "notes folder": r"C:\projects\notes",
        "group one": r"\1 and \g<0>",
    })
    out, _ = d.apply("open the temp folder, then the notes folder, then group one")
    assert out == r"open the C:\temp, then the C:\projects\notes, then \1 and \g<0>"


def test_profiles_by_app_and_title():
    assert profile_for("slack.exe").key == "chat"
    assert profile_for("OUTLOOK.EXE").key == "email"
    assert profile_for("Code.exe").key == "code"
    assert profile_for("WindowsTerminal.exe").key == "terminal"
    assert profile_for("brave.exe", "Inbox (3) - arnab@gmail.com - Gmail").key == "email"
    assert profile_for("chrome.exe", "general - Slack").key == "chat"
    assert profile_for("chrome.exe", "Some news site").key == "default"
    assert profile_for(None).key == "default"


# --- the LLM layer: when it runs, and what is rejected ------------------------------------
from localflow.cleanup.pipeline import CleanupPipeline, _strip_wrapping  # noqa: E402
from localflow.config import PostProcessConfig  # noqa: E402


class FakeProvider:
    def __init__(self, reply="", fail=False, truncated=False):
        self.reply, self.fail, self.truncated = reply, fail, truncated
        self.calls, self.prefills, self.budgets = [], [], []

    def complete(self, system, user, *, max_tokens=512, temperature=0.0):
        from localflow.llm.providers import Completion

        self.calls.append((system, user))
        self.budgets.append(max_tokens)
        if self.fail:
            raise RuntimeError("model is down")
        return Completion(self.reply or user, 12.0, truncated=self.truncated)

    def prefill(self, system, user):
        self.prefills.append(user)


def pipe(provider=None, **kw):
    return CleanupPipeline(PostProcessConfig(llm_cleanup=provider is not None, **kw), provider)


def test_short_utterances_skip_the_model_unless_they_carry_cues():
    p = pipe(FakeProvider())
    assert not p.needs_llm("thanks a lot")               # 3 words, nothing to fix
    assert p.needs_llm("Monday, no wait, Tuesday")       # self-correction cue
    assert p.needs_llm("see you at nine fifteen")        # number words
    assert p.needs_llm("one two three four five six seven")  # long enough on its own
    assert not pipe().needs_llm("anything at all")       # no provider: never


def test_llm_output_is_used_when_it_looks_like_an_edit():
    prov = FakeProvider("Send it Tuesday.")
    res = pipe(prov).process("send it Monday, no wait, Tuesday", app="slack.exe")
    assert res.text == "Send it Tuesday." and res.used_llm and res.llm_rejected is None
    assert "chat" in prov.calls[0][0].lower(), "the app's style profile reaches the prompt"


def test_guard_rejects_assistant_speak_and_keeps_the_rules_output():
    prov = FakeProvider("Sure! Here's the cleaned text: Send it Tuesday.")
    res = pipe(prov).process("send it Monday, no wait, Tuesday")
    assert not res.used_llm and res.llm_rejected == "assistant-speak"
    assert res.text == res.rules_text


def test_guard_rejects_a_model_that_answers_the_question():
    long_answer = "North Korea has a population of about 26 million people according to recent estimates today"
    prov = FakeProvider(long_answer)
    res = pipe(prov).process("what is the population of North Korea")
    assert not res.used_llm and res.llm_rejected.startswith("length")


def test_model_failure_falls_back_to_rules():
    prov = FakeProvider(fail=True)
    res = pipe(prov).process("send it Monday, no wait, Tuesday")
    assert not res.used_llm and res.llm_rejected.startswith("error:")
    assert res.text == res.rules_text


def test_prefill_passes_the_partial_transcript():
    prov = FakeProvider()
    pipe(prov).prefill("we should ship it on", app="slack.exe")
    assert prov.prefills == ["we should ship it on"]


def test_strip_wrapping_cleans_small_model_habits():
    assert _strip_wrapping("```\nHello there\n```") == "Hello there"
    assert _strip_wrapping("<think>hmm</think>Hello there") == "Hello there"
    assert _strip_wrapping("line one   \nline two  ") == "line one\nline two"
    assert _strip_wrapping("One small thing — could you add it?") == "One small thing, could you add it?"


def test_spoken_punctuation_is_opt_in():
    assert pipe().rules("a long period of time") == "A long period of time"
    assert pipe(spoken_punctuation=True).rules("hello comma world period") == "Hello, world."


def test_guard_rejects_silent_word_deletion():
    """Measured on a real dictation: the model dropped 'She said' from the front."""
    prov = FakeProvider("The meeting was productive, but we still need a decision on pricing.")
    res = pipe(prov).process("She said the meeting was productive, but we still need a decision on pricing")
    assert not res.used_llm and res.llm_rejected.startswith("dropped words")


def test_guard_allows_deletions_when_the_speaker_corrected_themselves():
    prov = FakeProvider("Let's go with the green one.")
    res = pipe(prov).process("let's go with the blue one, scratch that, the green one")
    assert res.used_llm, res.llm_rejected


def test_guard_allows_filler_removal_and_rewording():
    prov = FakeProvider("So I think we should ship it on Friday.")
    res = pipe(prov).process("um so I think, uh, we should basically ship it on Friday")
    assert res.used_llm, res.llm_rejected


def test_guard_allows_number_words_becoming_digits():
    """The guard must not read "twelfth" -> "12th" as a deleted word."""
    prov = FakeProvider("I'll be in Bangalore from the 12th to the 19th of October.")
    res = pipe(prov).process("I'll be in Bangalore from the twelfth to the nineteenth of October")
    assert res.used_llm, res.llm_rejected
    prov2 = FakeProvider("Set the timeout to 30 seconds and retry three times before giving up.")
    res2 = pipe(prov2).process("set the timeout to thirty seconds and retry three times before giving up")
    assert res2.used_llm, res2.llm_rejected


def test_generation_is_capped_at_the_length_the_guard_would_reject():
    """A model that answers instead of editing used to run for 1.4 s before the guard threw
    the output away. It may not spend more tokens than an acceptable edit could need."""
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig

    provider = FakeProvider()
    pipe = CleanupPipeline(PostProcessConfig(llm_cleanup=True, llm_max_tokens=400), provider)

    short = "The concept of sex slaves in Islam."
    long = " ".join(["word"] * 200)
    assert pipe.budget(short) < 100, "a seven-word edit does not need a hundred tokens"
    assert pipe.budget(short) >= 48, "short utterances still need room"
    assert pipe.budget(long) == 400, "the configured ceiling still applies"
    # The cap must leave room for every edit the guard would accept.
    assert pipe.budget(short) > len(short.split()) * 1.6


def test_output_that_hit_the_cap_is_rejected_rather_than_used():
    """Truncated output is a fragment. Its length ratio can land inside the guard's window by
    accident, so it has to be refused on its own terms or a half sentence gets typed."""
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig

    said = "i went to the shop and bought some milk and then i walked home again"
    provider = FakeProvider(reply="I went to the shop and bought some milk and then I", truncated=True)
    pipe = CleanupPipeline(PostProcessConfig(llm_cleanup=True), provider)
    result = pipe.process(said)
    assert result.llm_rejected == "truncated"
    assert not result.used_llm
    assert result.text == result.rules_text


def test_fillers_go_but_the_same_letters_meaning_something_stay():
    """"ER" and "mm" were removed as hesitations: "take him to the ER" lost its last word and
    "a 5 mm screw" became "a 5 screw"."""
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig

    rules = CleanupPipeline(PostProcessConfig(llm_cleanup=False)).rules
    assert rules("um so I think, uh, we should er ship it") == "So I think, we should ship it"
    assert rules("Er, hmm, take him to the ER now") == "Take him to the ER now"
    assert rules("use a 5 mm screw and a 12mm bolt, mm, maybe") == "Use a 5 mm screw and a 12mm bolt, maybe"
    assert rules("the UM campus") == "The UM campus"
