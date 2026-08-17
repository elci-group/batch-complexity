//! Deterministic complexity scoring for an inference batch.
//!
//! No model calls, no randomness — given the same input this always produces
//! the same score. The goal is a cheap pre-flight signal for whether a batch
//! is likely to need a stronger (and more expensive/quota-constrained) model
//! to extract its findings reliably, versus one that a smaller model can
//! handle. It is intentionally conservative: signals that correlate with
//! *more distinct, concrete, technical content* push the score up, since that
//! is what strains structured extraction (see the `dreamseq-analysis-v1`
//! taxonomy categories this is calibrated against).

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// A single chat message, matching the shape dreamsequence-api's
/// `/api/v1/inference` already accepts and dreamseq already builds.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Route to the smaller/cheaper model.
    Light,
    /// Route to the larger model.
    Heavy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signals {
    pub char_count: usize,
    pub length_score: f64,
    pub keyword_hits: usize,
    pub keyword_categories_touched: usize,
    pub keyword_score: f64,
    pub repeated_ngrams: usize,
    pub repetition_score: f64,
    pub technical_tokens: usize,
    pub technical_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityScore {
    /// 0.0 (trivially simple) to 100.0 (dense, technical, multi-signal).
    pub score: f64,
    pub tier: Tier,
    pub signals: Signals,
}

/// Score at or above this routes to the Heavy tier. Calibrated against the
/// initial benchmark corpus (see tests/); revisit as more real batches are
/// scored and their actual extraction outcomes are known.
pub const HEAVY_THRESHOLD: f64 = 32.0;

// Keyword sets loosely mirroring resources/analysis-taxonomy.txt's eight
// categories. Deliberately simple substring/regex matches, not NLP — the
// point is a cheap, explainable, reproducible signal, not full understanding.
static ERROR_CODE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(HTTP\s?)?[45]\d{2}\b").unwrap());
static BACKTICK_TOKEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"`[^`]{1,80}`").unwrap());
static PATH_TOKEN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b[\w-]+(?:/[\w.-]+){1,}\b|\b[\w-]+\.[a-zA-Z]{1,5}\b").unwrap());
static SEGMENT_MARKER: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[segment\s+\d+/\d+\]").unwrap());

struct KeywordCategory {
    #[allow(dead_code)]
    name: &'static str,
    words: &'static [&'static str],
}

const KEYWORD_CATEGORIES: &[KeywordCategory] = &[
    KeywordCategory {
        name: "model_failures",
        words: &[
            "refused",
            "refusal",
            "overloaded",
            "safety",
            "hallucinat",
            "provider error",
        ],
    },
    KeywordCategory {
        name: "harness_friction",
        words: &[
            "crashed",
            "hang",
            "hung",
            "socket hang up",
            "corrupt",
            "froze",
            "frozen",
        ],
    },
    KeywordCategory {
        name: "missing_tooling",
        words: &[
            "does not exist",
            "no tool",
            "missing tool",
            "not available",
            "unavailable",
            "no adapter",
        ],
    },
    KeywordCategory {
        name: "workflow_bottlenecks",
        words: &[
            "manual",
            "manually",
            "copy-paste",
            "copied the log",
            "waited for",
            "delay",
        ],
    },
    KeywordCategory {
        name: "repeated_commands",
        words: &[
            "ran again",
            "re-ran",
            "rerun",
            "retried",
            "repeated the command",
        ],
    },
    KeywordCategory {
        name: "repeated_prompts",
        words: &[
            "already said",
            "already told",
            "restate",
            "re-stated",
            "again:",
            "for the third time",
            "for the fourth time",
        ],
    },
    KeywordCategory {
        name: "context_loss",
        words: &[
            "compact",
            "compaction",
            "lost context",
            "re-asked",
            "which repository",
            "rediscover",
        ],
    },
    KeywordCategory {
        name: "automation_opportunities",
        words: &[
            "could automate",
            "pre-commit hook",
            "estimated",
            "would save",
            "automation",
        ],
    },
];

/// Extracts complexity-relevant content: user/assistant turns only. The
/// system message (the taxonomy policy prompt itself) carries no signal
/// about *this* batch and would bias every score upward identically.
fn relevant_content(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn count_repeated_ngrams(text: &str, n: usize) -> usize {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        return 0;
    }
    let mut counts = std::collections::HashMap::new();
    for window in words.windows(n) {
        *counts.entry(window.join(" ")).or_insert(0usize) += 1;
    }
    counts.values().filter(|&&c| c >= 2).count()
}

/// Saturating scale: linear up to `knee`, then logarithmic — a batch of
/// 20,000 chars is not 10x as complex as one of 2,000, but it is more.
fn saturating_scale(value: f64, knee: f64) -> f64 {
    if value <= knee {
        value / knee
    } else {
        1.0 + (value / knee).ln()
    }
}

pub fn score(messages: &[Message]) -> ComplexityScore {
    let text = relevant_content(messages);
    let char_count = text.chars().count();

    let length_score = saturating_scale(char_count as f64, 800.0).min(3.0) * 10.0; // 0..~30

    let lower = text.to_lowercase();
    let mut keyword_hits = 0usize;
    let mut categories_touched = 0usize;
    for category in KEYWORD_CATEGORIES {
        let mut hit_in_category = false;
        for word in category.words {
            let count = lower.matches(word).count();
            if count > 0 {
                keyword_hits += count;
                hit_in_category = true;
            }
        }
        if hit_in_category {
            categories_touched += 1;
        }
    }
    let error_code_hits = ERROR_CODE.find_iter(&text).count();
    keyword_hits += error_code_hits;
    let keyword_score = (keyword_hits as f64 * 4.0 + categories_touched as f64 * 6.0).min(40.0);

    let repeated_ngrams = count_repeated_ngrams(&text, 3);
    let repetition_score = (repeated_ngrams as f64 * 5.0).min(15.0);

    let technical_tokens = BACKTICK_TOKEN.find_iter(&text).count()
        + PATH_TOKEN.find_iter(&text).count()
        + SEGMENT_MARKER.find_iter(&text).count();
    let technical_score = (technical_tokens as f64 * 1.5).min(15.0);

    let total = (length_score + keyword_score + repetition_score + technical_score).min(100.0);
    let tier = if total >= HEAVY_THRESHOLD {
        Tier::Heavy
    } else {
        Tier::Light
    };

    ComplexityScore {
        score: (total * 100.0).round() / 100.0,
        tier,
        signals: Signals {
            char_count,
            length_score: round2(length_score),
            keyword_hits,
            keyword_categories_touched: categories_touched,
            keyword_score: round2(keyword_score),
            repeated_ngrams,
            repetition_score: round2(repetition_score),
            technical_tokens,
            technical_score: round2(technical_score),
        },
    }
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}
