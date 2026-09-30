//! The user's own words, to take to another PC or keep safe: the dictionary, the terms spelt by
//! sound, snippets, per-app rules and house style. One JSON file (Hub > Help), merged on import:
//! what is new is added, and where the user already has an entry of the same name, theirs stays.
//!
//! The first four live in the engine's config (the clean-up pipeline uses them) and the app
//! rules in the shell's settings; the file holds both, so it is one file for the user.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::settings::AppRule;

/// What says "this is a LocalFlow words file" - nothing else is imported.
pub const KIND: &str = "localflow-words";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Words {
    pub kind: String,
    pub version: u32,
    /// Exact replacements, as heard -> as meant.
    pub dictionary: BTreeMap<String, String>,
    /// Correct spellings, matched by sound.
    pub dictionary_terms: Vec<String>,
    /// Trigger phrase -> text.
    pub snippets: BTreeMap<String, String>,
    /// Executable name -> rule.
    pub app_rules: BTreeMap<String, AppRule>,
    /// The clean-up model's house style.
    pub custom_instructions: String,
}

impl Words {
    /// From the engine's config (as the file holds it) and the shell's settings.
    pub fn gather(engine_config: &Value, app_rules: &BTreeMap<String, AppRule>) -> Words {
        let pp = engine_config.get("postprocess").cloned().unwrap_or(Value::Null);
        let map = |key: &str| -> BTreeMap<String, String> {
            pp.get(key).and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default()
        };
        Words {
            kind: KIND.into(),
            version: VERSION,
            dictionary: map("dictionary"),
            dictionary_terms: pp.get("dictionary_terms").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default(),
            snippets: map("snippets"),
            app_rules: app_rules.clone(),
            custom_instructions: pp.get("custom_instructions").and_then(Value::as_str).unwrap_or("").to_owned(),
        }
    }

    /// A file's contents, if it is one of these.
    pub fn parse(text: &str) -> Result<Words, String> {
        let words: Words = serde_json::from_str(text.trim_start_matches('\u{feff}'))
            .map_err(|e| format!("this isn't a LocalFlow words file ({e})"))?;
        if words.kind != KIND {
            return Err("this isn't a LocalFlow words file (it doesn't say it is one)".into());
        }
        if words.version > VERSION {
            return Err(format!("this file is from a newer LocalFlow (version {}); update LocalFlow to import it", words.version));
        }
        Ok(words)
    }

    /// The engine's part, as a `settings.set` postprocess patch.
    pub fn postprocess_patch(&self) -> Value {
        json!({
            "dictionary": self.dictionary,
            "dictionary_terms": self.dictionary_terms,
            "snippets": self.snippets,
            "custom_instructions": self.custom_instructions,
        })
    }
}

/// What an import did, for the Hub to say.
#[derive(Debug, Default, PartialEq, Serialize)]
pub struct Merged {
    pub added: usize,
    /// Entries the user already had under the same name, kept as they were.
    pub kept: usize,
    pub style: &'static str,
}

/// `mine` with what `theirs` adds. Same name, different content: mine stays.
pub fn merge(mine: &Words, theirs: &Words) -> (Words, Merged) {
    let mut out = mine.clone();
    let mut m = Merged::default();
    fn add<V: Clone + PartialEq>(into: &mut BTreeMap<String, V>, from: &BTreeMap<String, V>, m: &mut Merged) {
        for (k, v) in from {
            match into.get(k) {
                None => {
                    into.insert(k.clone(), v.clone());
                    m.added += 1;
                }
                Some(existing) if existing != v => m.kept += 1,
                Some(_) => {}
            }
        }
    }
    add(&mut out.dictionary, &theirs.dictionary, &mut m);
    add(&mut out.snippets, &theirs.snippets, &mut m);
    add(&mut out.app_rules, &theirs.app_rules, &mut m);
    for term in &theirs.dictionary_terms {
        if !out.dictionary_terms.iter().any(|t| t.eq_ignore_ascii_case(term)) {
            out.dictionary_terms.push(term.clone());
            m.added += 1;
        }
    }
    // House style: the lines not already there go after mine.
    let new_lines: Vec<&str> = theirs
        .custom_instructions
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !mine.custom_instructions.lines().any(|x| x.trim() == *l))
        .collect();
    m.style = if new_lines.is_empty() {
        "unchanged"
    } else if mine.custom_instructions.trim().is_empty() {
        out.custom_instructions = new_lines.join("\n");
        "added"
    } else {
        out.custom_instructions = format!("{}\n{}", mine.custom_instructions.trim_end(), new_lines.join("\n"));
        "extended"
    };
    (out, m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(dict: &[(&str, &str)], terms: &[&str], style: &str) -> Words {
        Words {
            kind: KIND.into(),
            version: VERSION,
            dictionary: dict.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
            dictionary_terms: terms.iter().map(|t| t.to_string()).collect(),
            custom_instructions: style.into(),
            ..Words::default()
        }
    }

    #[test]
    fn an_import_adds_and_never_overwrites() {
        let mine = words(&[("kwen", "Qwen"), ("arnub", "Arnab")], &["Okonkwo"], "British spelling.");
        let theirs = words(&[("kwen", "Kwen"), ("priya", "Priya")], &["okonkwo", "Parakeet"], "British spelling.\nNo emoji.");
        let (out, m) = merge(&mine, &theirs);
        assert_eq!(out.dictionary["kwen"], "Qwen", "mine stays");
        assert_eq!(out.dictionary["priya"], "Priya");
        assert_eq!(out.dictionary_terms, ["Okonkwo", "Parakeet"], "a term differing only in case is the same term");
        assert_eq!(out.custom_instructions, "British spelling.\nNo emoji.");
        assert_eq!(m, Merged { added: 2, kept: 1, style: "extended" });
        // Importing the same file again changes nothing.
        let (again, m2) = merge(&out, &theirs);
        assert_eq!((again, m2.added, m2.style), (out, 0, "unchanged"));
    }

    #[test]
    fn only_a_words_file_is_taken() {
        assert!(Words::parse(r#"{"stt": {"model": "x"}}"#).unwrap_err().contains("isn't a LocalFlow words file"));
        assert!(Words::parse("not json").is_err());
        let newer = format!(r#"{{"kind": "{KIND}", "version": 9}}"#);
        assert!(Words::parse(&newer).unwrap_err().contains("newer LocalFlow"));
        let ok = format!("\u{feff}{{\"kind\": \"{KIND}\", \"version\": 1, \"snippets\": {{\"sig\": \"Best, A.\"}}}}");
        assert_eq!(Words::parse(&ok).unwrap().snippets["sig"], "Best, A.");
    }

    #[test]
    fn gathered_from_where_each_part_lives() {
        let cfg = json!({"postprocess": {"dictionary": {"kwen": "Qwen"}, "dictionary_terms": ["Okonkwo"],
            "snippets": {"sig": "Best"}, "custom_instructions": "Short.", "llm_api_key": "sk-x"}});
        let mut rules = BTreeMap::new();
        rules.insert("slack.exe".to_owned(), AppRule { auto_send: true, ..AppRule::default() });
        let w = Words::gather(&cfg, &rules);
        assert_eq!((w.dictionary.len(), w.snippets.len(), w.app_rules.len()), (1, 1, 1));
        let text = serde_json::to_string(&w).unwrap();
        assert!(!text.contains("sk-x"), "no API key in a file people move around");
    }
}
