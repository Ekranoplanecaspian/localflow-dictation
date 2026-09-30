from dataclasses import replace

from localflow.config import PostProcessConfig
from localflow.validate import MAX_INSTRUCTION_CHARS, check_postprocess

OLD = PostProcessConfig(llm_provider="openai", llm_url="https://api.example.com", llm_api_key="sk-old",
                        dictionary_terms=["Arnab"], snippets={"my email": "a@example.com"})


def check(**changes):
    return check_postprocess(replace(OLD, **changes), OLD)


def test_an_empty_snippet_trigger_is_never_kept():
    """It matched between every two words and was inserted all through every dictation."""
    pp, _ = check(snippets={"": "SPAM", "  sig  ": "Arnab Arya"})
    assert pp.snippets == {"sig": "Arnab Arya"}


def test_words_are_tidied_and_duplicates_dropped():
    pp, problems = check(dictionary_terms=["  Okonkwo ", "okonkwo", "", "Priya  Rao", 7])
    assert pp.dictionary_terms == ["Okonkwo", "Priya Rao"] and problems == []


def test_a_pasted_api_key_loses_its_newline_and_an_address_its_slash():
    pp, problems = check(llm_api_key="sk-new\r\n", llm_url=" https://api.groq.com/openai/ ")
    assert pp.llm_api_key == "sk-new" and pp.llm_url == "https://api.groq.com/openai" and problems == []


def test_unusable_values_keep_the_old_one_and_say_why():
    pp, problems = check(llm_url="api.example.com", llm_provider="gemini", llm_min_words=-3, llm_cleanup="yes")
    assert (pp.llm_url, pp.llm_provider, pp.llm_min_words, pp.llm_cleanup) == (
        OLD.llm_url, OLD.llm_provider, OLD.llm_min_words, OLD.llm_cleanup)
    assert len(problems) == 4
    assert any("http" in p for p in problems)


def test_instructions_too_long_for_the_model_are_refused():
    pp, problems = check(custom_instructions="x" * (MAX_INSTRUCTION_CHARS + 1))
    assert pp.custom_instructions == OLD.custom_instructions and problems


def test_good_settings_pass_untouched():
    pp, problems = check(custom_instructions="Use British spelling.", llm_min_words=4)
    assert problems == [] and pp.custom_instructions == "Use British spelling." and pp.llm_min_words == 4
