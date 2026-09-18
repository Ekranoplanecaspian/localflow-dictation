from localflow.bench import edit_distance, normalize


def test_normalize_strips_case_and_punctuation():
    assert normalize("Hello, World! It's late.") == ["hello", "world", "it's", "late"]
    assert normalize("candle-light") == ["candle", "light"]


def test_edit_distance_counts_substitutions_insertions_deletions():
    ref = normalize("the quick brown fox")
    assert edit_distance(ref, normalize("the quick brown fox")) == 0
    assert edit_distance(ref, normalize("the quick brown dog")) == 1  # substitution
    assert edit_distance(ref, normalize("the quick brown")) == 1  # deletion
    assert edit_distance(ref, normalize("the very quick brown fox")) == 1  # insertion
    assert edit_distance(ref, []) == 4


def test_punctuated_hypothesis_scores_zero():
    ref = normalize("THUS IT IS THAT THE HONOUR OF THREE IS SAVED")
    hyp = normalize("Thus it is that the honour of three is saved.")
    assert edit_distance(ref, hyp) == 0


def test_numbers_words_and_digits_are_equal():
    assert normalize("three thirty") == normalize("3:30") == ["3", "30"]
    assert normalize("nine fifteen") == normalize("9:15")
    assert normalize("thirty seconds") == normalize("30 seconds")
    assert normalize("two eggs") == normalize("2 eggs")
    assert normalize("two hundred fifty grams") == normalize("250 grams") == ["250", "g"]
    assert normalize("March 1st") == normalize("March 1") == ["march", "1"]
    assert normalize("32 gigabytes") == normalize("32GB") == ["32", "gb"]


def test_digit_by_digit_runs_join_like_invoice_numbers():
    assert normalize("four four seven one") == ["4471"]
    assert normalize("it's 4471, not 4417") == ["it's", "4471", "not", "4417"]
    assert normalize("ends in eight eight two three") == ["ends", "in", "8823"]


def test_compound_spelling_variants_are_equal():
    assert normalize("the standup") == normalize("the stand up")
    assert normalize("clean-up model") == normalize("cleanup model")


def test_own_set_formatting_cases_score_zero():
    pairs = [
        ("Let's schedule the design review for three thirty on Thursday afternoon.",
         "Let's schedule the design review for 3:30 on Thursday afternoon."),
        ("Can we move the standup to 9:15 so the London team can join?",
         "Can we move the stand up to nine fifteen so the London team can join?"),
        ("The new laptop has 32 gigabytes of RAM and an RTX 4060 graphics card.",
         "The new laptop has 32GB of RAM and an RTX 4060 graphics card."),
        ("Set the timeout to 30 seconds and retry three times before giving up.",
         "Set the timeout to thirty seconds and retry three times before giving up."),
    ]
    for ref, hyp in pairs:
        assert edit_distance(normalize(ref), normalize(hyp)) == 0, (ref, hyp)
