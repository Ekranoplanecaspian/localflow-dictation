//! Dictionary suggestions, learned from what clean-up keeps correcting.
//!
//! Every history entry holds both halves of the story: `raw`, what the speech model heard, and
//! `text`, what was actually typed after clean-up. Where those disagree on a single word, the
//! same way, over and over, that is a term the speech model cannot spell and the user is
//! quietly re-fixing - "kwen" for "Qwen", "parakeet" for "Parakeet". Those are exactly the
//! entries the personal dictionary exists to hold, and asking someone to notice the pattern and
//! type it in themselves is asking them to do arithmetic the machine already has the numbers
//! for.
//!
//! This only ever *suggests*. Nothing is added without the user saying so: a wrong dictionary
//! entry silently rewrites a word in every future dictation, which is far worse than not having
//! the entry at all.

use std::collections::HashMap;

use serde::Serialize;

/// How many times a correction must repeat before it is worth mentioning. Two could be one
/// phrase dictated twice; three is a habit.
const MIN_SEEN: u32 = 3;
/// Words shorter than this produce noise - "a" for "the", "in" for "on" - and are never the
/// kind of proper noun a dictionary entry is for.
const MIN_LEN: usize = 4;
/// Corrections found in more entries than this are almost certainly the model rephrasing
/// rather than a spelling the user wants fixed.
const MAX_SUGGESTIONS: usize = 12;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Suggestion {
    /// What the speech model keeps hearing.
    pub from: String,
    /// What it keeps being corrected to.
    pub to: String,
    /// How many dictations this happened in.
    pub seen: u32,
}

/// Split into lower-case words, keeping apostrophes and hyphens inside a word.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '-'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// The same, but keeping the original spelling, so a suggestion can propose real capitals.
fn words_cased(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '-'))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// One-for-one word substitutions between two sequences.
///
/// A longest-common-subsequence walk rather than a positional comparison: clean-up inserts and
/// deletes words as well as changing them, and a positional diff would report every word after
/// a removed "um" as a substitution.
fn substitutions(
    from: &[String],
    from_cased: &[String],
    to: &[String],
    to_cased: &[String],
) -> Vec<(String, String, usize)> {
    let (n, m) = (from.len(), to.len());
    if n == 0 || m == 0 || n > 400 || m > 400 {
        return Vec::new();
    }
    // lcs[i][j] = length of the longest common subsequence of from[i..] and to[j..]
    let mut lcs = vec![vec![0u16; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if from[i] == to[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }

    let mut pairs = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if from[i] == to[j] {
            // The same word, but clean-up may have changed its case. That is invisible to the
            // comparison above, which works on lower-case forms - and it is the most common
            // correction of all, because a speech model that has never seen a name writes it
            // in lower case every time.
            let was = from_cased.get(i).map(String::as_str).unwrap_or("");
            let now = to_cased.get(j).map(String::as_str).unwrap_or("");
            if was != now && was.to_lowercase() == now.to_lowercase() {
                pairs.push((from[i].clone(), now.to_owned(), j));
            }
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            // from[i] was dropped. If exactly one word was dropped here and exactly one
            // inserted opposite it, that is a substitution rather than an edit.
            let dropped = from[i].clone();
            i += 1;
            if i < n && j < m && from[i] == to[j] {
                // nothing took its place: a deletion, not a substitution
                let _ = dropped;
            } else if j < m && i <= n {
                let inserted = to[j].clone();
                let cased = to_cased.get(j).cloned().unwrap_or_else(|| inserted.clone());
                pairs.push((dropped, cased, j));
                j += 1;
            }
        } else {
            j += 1;
        }
    }
    pairs
}

/// Levenshtein distance, on characters.
fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Does this correction look like a name or a term rather than the model rewording?
///
/// `at_start` is whether the corrected word opened the dictation, which decides what a pure
/// capitalisation means.
fn worth_suggesting(from: &str, to: &str, at_start: bool) -> bool {
    let to_lower = to.to_lowercase();
    if from.len() < MIN_LEN || to_lower.len() < MIN_LEN {
        return false;
    }
    if to.chars().any(|c| c.is_ascii_digit()) || from.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    if from == to_lower {
        // Only the capital changed. In the middle of a sentence that is a proper noun the
        // speech model does not know - exactly what the dictionary is for. At the very start it
        // is just clean-up capitalising the first word, and an entry made from that would then
        // capitalise the word everywhere else, where it is wrong.
        return !at_start;
    }
    // Otherwise the two must be plausibly the same word misheard rather than one word swapped
    // for another. Distance rather than a shared first letter: a mishearing very often changes
    // the first sound - "kwen" for "Qwen" is the case this exists to catch.
    let allowed = (to_lower.len() / 3).max(1);
    let d = distance(from, &to_lower);
    d > 0 && d <= allowed
}

/// Corrections the user keeps making, most frequent first.
///
/// `known` are the terms already in the dictionary; suggesting those again would be noise.
pub fn suggestions(entries: &[crate::history::Entry], known: &[String]) -> Vec<Suggestion> {
    let known: Vec<String> = known.iter().map(|k| k.to_lowercase()).collect();
    let mut counts: HashMap<(String, String), u32> = HashMap::new();

    for entry in entries {
        if entry.raw.is_empty() || entry.text.is_empty() {
            continue;
        }
        let from = words(&entry.raw);
        let from_cased = words_cased(&entry.raw);
        let to = words(&entry.text);
        let to_cased = words_cased(&entry.text);
        // One entry votes once for a given correction, so a word repeated within a single
        // dictation cannot reach the threshold on its own.
        let mut seen_here: Vec<(String, String)> = Vec::new();
        for (a, b, at) in substitutions(&from, &from_cased, &to, &to_cased) {
            if !worth_suggesting(&a, &b, at == 0) {
                continue;
            }
            if known.contains(&b.to_lowercase()) {
                continue;
            }
            let pair = (a, b);
            if !seen_here.contains(&pair) {
                seen_here.push(pair);
            }
        }
        for pair in seen_here {
            *counts.entry(pair).or_insert(0) += 1;
        }
    }

    let mut out: Vec<Suggestion> = counts
        .into_iter()
        .filter(|(_, seen)| *seen >= MIN_SEEN)
        .map(|((from, to), seen)| Suggestion { from, to, seen })
        .collect();
    out.sort_by(|a, b| b.seen.cmp(&a.seen).then(a.from.cmp(&b.from)));
    out.truncate(MAX_SUGGESTIONS);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::Entry;

    fn entry(raw: &str, text: &str) -> Entry {
        Entry {
            at: 0,
            raw: raw.into(),
            text: text.into(),
            app: "brave.exe".into(),
            words: 0,
            audio_s: 0.0,
            ms: 0,
            used_llm: true,
        }
    }

    #[test]
    fn a_name_corrected_again_and_again_is_suggested() {
        let entries: Vec<Entry> = (0..3)
            .map(|_| entry("we shipped kwen to production", "We shipped Qwen to production"))
            .collect();
        let out = suggestions(&entries, &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].from, "kwen");
        assert_eq!(out[0].to, "Qwen");
        assert_eq!(out[0].seen, 3);
    }

    #[test]
    fn a_correction_seen_once_or_twice_is_not_yet_a_habit() {
        let entries: Vec<Entry> = (0..2)
            .map(|_| entry("we shipped kwen today", "We shipped Qwen today"))
            .collect();
        assert!(suggestions(&entries, &[]).is_empty());
    }

    /// Clean-up capitalises the first word of every take. A dictionary entry made from that
    /// would then capitalise the word in the middle of sentences, where it is wrong.
    #[test]
    fn a_plain_capital_is_never_suggested() {
        let entries: Vec<Entry> =
            (0..5).map(|_| entry("shipping it today", "Shipping it today")).collect();
        assert!(suggestions(&entries, &[]).is_empty());
    }

    #[test]
    fn terms_already_in_the_dictionary_are_not_suggested_again() {
        let entries: Vec<Entry> = (0..4)
            .map(|_| entry("we shipped kwen to production", "We shipped Qwen to production"))
            .collect();
        assert!(suggestions(&entries, &["qwen".into()]).is_empty());
    }

    /// The whole point of the LCS walk: clean-up removes fillers, and a positional comparison
    /// would then report every following word as a substitution.
    #[test]
    fn removing_a_filler_does_not_look_like_a_correction() {
        let entries: Vec<Entry> = (0..5)
            .map(|_| entry("um we should ship this today", "We should ship this today"))
            .collect();
        assert!(suggestions(&entries, &[]).is_empty());
    }

    #[test]
    fn one_word_swapped_for_an_unrelated_one_is_not_a_spelling() {
        let entries: Vec<Entry> =
            (0..5).map(|_| entry("send it monday morning", "send it Thursday morning")).collect();
        assert!(suggestions(&entries, &[]).is_empty(), "different word, not a misspelling");
    }

    #[test]
    fn very_short_words_are_ignored() {
        let entries: Vec<Entry> =
            (0..5).map(|_| entry("put it on the shelf", "put it in the shelf")).collect();
        assert!(suggestions(&entries, &[]).is_empty());
    }

    #[test]
    fn the_most_frequent_corrections_come_first() {
        let mut entries: Vec<Entry> = (0..5)
            .map(|_| entry("the parakeet model", "the Parakeet model"))
            .collect();
        entries.extend(
            (0..3).map(|_| entry("we shipped kwen to production", "We shipped Qwen to production")),
        );
        let out = suggestions(&entries, &[]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].to, "Parakeet");
        assert_eq!(out[1].to, "Qwen");
    }

    #[test]
    fn an_empty_history_suggests_nothing() {
        assert!(suggestions(&[], &[]).is_empty());
    }
}
