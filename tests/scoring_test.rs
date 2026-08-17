use batch_complexity::{Message, Tier, score};

fn user_message(content: &str) -> Vec<Message> {
    vec![Message {
        role: "user".to_string(),
        content: content.to_string(),
    }]
}

// Same three samples used in the manual Groq benchmark this tool is meant to
// replace the guesswork behind — kept in sync deliberately as a regression
// anchor between the two.
const SPARSE_CLEAN: &str = "[segment 2/9] User asked agent to add a null check to `parseConfig()` in \
config.rs. Agent read the file, added the check, ran `cargo test config::`, all 4 \
tests passed on the first attempt. User replied \"looks good, thanks\" and the session \
ended. No errors, retries, or repeated instructions observed in this segment.";

const RICH_SIGNAL: &str = "[segment 4/9] Agent invoked `npm test` 6 times in the last 12 minutes with \
identical output each time (142 passing, 3 failing in auth.spec.ts). User message: \
\"you already ran this, just fix the failing ones instead of re-running the suite.\" \
Agent then attempted to call a tool `get_ci_status` which does not exist in this \
harness's tool registry; fell back to asking the user to paste CI logs manually. \
Later, agent's context was compacted after 40 turns; on the next turn it re-asked \
\"which repository are we working in?\" despite this being established at turn 2. \
Provider response at 14:02:11 returned HTTP 529 (overloaded), retried twice before \
succeeding with claude-3-5-sonnet.";

const MULTI_ISSUE: &str = "[segment 7/9] Session spanned three repositories (api, web, worker). Agent \
repeated the exact instruction \"always run lint before committing\" from the user \
four separate times across turns 12, 34, 51, and 88 — each time the agent had skipped \
linting. Harness's MCP browser-preview tool crashed twice with \"socket hang up\", \
losing the agent's in-progress screenshot review state both times and forcing a \
manual restart. Agent proposed writing a pre-commit hook to auto-run lint, estimating \
it would save roughly 15 minutes per week per engineer. Separately, agent flagged \
that no `changelog-entry` tool exists for this repo despite three PRs in the last \
week needing manual changelog edits. One inference call to the underlying model \
failed with a content-safety refusal on a benign refactor request, requiring the \
user to rephrase before the agent would proceed.";

#[test]
fn sparse_clean_batch_scores_light() {
    let result = score(&user_message(SPARSE_CLEAN));
    assert_eq!(
        result.tier,
        Tier::Light,
        "sparse batch should not need the heavy model, got score {}",
        result.score
    );
}

#[test]
fn rich_signal_batch_scores_heavy() {
    let result = score(&user_message(RICH_SIGNAL));
    assert_eq!(
        result.tier,
        Tier::Heavy,
        "dense multi-signal batch should route heavy, got score {}",
        result.score
    );
}

#[test]
fn multi_issue_batch_scores_heavy() {
    let result = score(&user_message(MULTI_ISSUE));
    assert_eq!(
        result.tier,
        Tier::Heavy,
        "multi-category batch should route heavy, got score {}",
        result.score
    );
}

#[test]
fn rich_and_multi_batches_score_higher_than_sparse() {
    let sparse = score(&user_message(SPARSE_CLEAN));
    let rich = score(&user_message(RICH_SIGNAL));
    let multi = score(&user_message(MULTI_ISSUE));
    assert!(rich.score > sparse.score);
    assert!(multi.score > sparse.score);
}

#[test]
fn empty_content_scores_minimally_and_stays_light() {
    let result = score(&user_message(""));
    assert_eq!(result.tier, Tier::Light);
    assert_eq!(result.signals.char_count, 0);
}

#[test]
fn system_message_content_does_not_influence_score() {
    let long_system_prompt = "x ".repeat(5000);
    let with_system = vec![
        Message {
            role: "system".to_string(),
            content: long_system_prompt,
        },
        Message {
            role: "user".to_string(),
            content: SPARSE_CLEAN.to_string(),
        },
    ];
    let without_system = user_message(SPARSE_CLEAN);
    assert_eq!(score(&with_system).score, score(&without_system).score);
}

#[test]
fn scoring_is_deterministic_across_repeated_calls() {
    let first = score(&user_message(RICH_SIGNAL));
    let second = score(&user_message(RICH_SIGNAL));
    assert_eq!(first.score, second.score);
    assert_eq!(first.tier, second.tier);
}

#[test]
fn pure_repetition_is_detected() {
    let repeated =
        "run the build again. run the build again. run the build again. run the build again.";
    let result = score(&user_message(repeated));
    assert!(
        result.signals.repeated_ngrams > 0,
        "expected repeated n-grams to be detected"
    );
}

#[test]
fn very_long_batch_does_not_exceed_max_score() {
    let huge = RICH_SIGNAL.repeat(50);
    let result = score(&user_message(&huge));
    assert!(result.score <= 100.0);
}
