//! Basic tests for the fidelity check module.

use agent_comm::check::{
    fidelity_report, find_blind_violation, find_sep_violation, ConsumerFn, Violation,
};
use agent_comm::{Content, Conversation, Role, Turn};
use serde_json::json;

fn text(s: &str) -> Content {
    Content::Text { text: s.into(), cache_control: None }
}

/// Consumer that only looks at top-level text content (ignores tool_call args' object
/// key order); a "kernel-only" consumer that treats nf-equal messages identically.
struct TextOnlyConsumer;
impl ConsumerFn for TextOnlyConsumer {
    fn observe(&self, conv: &Conversation) -> Vec<u8> {
        let mut out = String::new();
        for t in &conv.turns {
            for c in &t.content {
                if let Content::Text { text, .. } = c {
                    out.push_str(text);
                }
            }
        }
        out.into_bytes()
    }
}

/// Consumer that observes raw JSON of the (un-normalized) conversation. Will distinguish
/// pairs that differ only by vendor-accidental structure (args key order, raw ids, …).
struct RawJsonConsumer;
impl ConsumerFn for RawJsonConsumer {
    fn observe(&self, conv: &Conversation) -> Vec<u8> {
        serde_json::to_vec(conv).unwrap()
    }
}

#[test]
fn find_blind_violation_returns_none_when_nf_aligned_with_responses() {
    let s1 = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![text("a")] }],
    };
    let s2 = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![text("b")] }],
    };
    // Distinct messages → distinct responses; no blind violation.
    assert!(find_blind_violation(&[s1, s2], &TextOnlyConsumer).is_none());
}

#[test]
fn find_blind_violation_detects_when_consumer_collapses_distinct_nfs() {
    // Two genuinely distinct conversations (different text) but a degenerate consumer
    // that ignores content entirely → consumer collapses them.
    let s1 = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![text("a")] }],
    };
    let s2 = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![text("b")] }],
    };
    let consumer = |_: &Conversation| vec![0u8];
    let v = find_blind_violation(&[s1, s2], &consumer);
    assert!(matches!(v, Some(Violation::Blind { i: 0, j: 1, .. })));
}

#[test]
fn find_sep_violation_detects_when_consumer_distinguishes_same_nf() {
    // Two conversations with the same normal form (R3 canonicalizes raw ids → "call_0")
    // but a consumer that looks at the raw (un-normalized) JSON sees the original
    // distinct ids and treats them as different.
    let mk = |raw_id: &str| Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![Content::ToolCall {
                id: raw_id.into(),
                name: "f".into(),
                args: json!({}),
            }],
        }],
    };
    let s1 = mk("raw_x");
    let s2 = mk("raw_y");
    assert_eq!(s1.normalize(), s2.normalize(), "precondition: nfs must agree");
    let v = find_sep_violation(&[s1, s2], &RawJsonConsumer);
    assert!(matches!(v, Some(Violation::Sep { i: 0, j: 1, .. })));
}

#[test]
fn find_sep_violation_returns_none_when_consumer_respects_normal_form() {
    let s1 = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![text("x")] }],
    };
    let s2 = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![text("x")] }],
    };
    assert!(find_sep_violation(&[s1, s2], &TextOnlyConsumer).is_none());
}

#[test]
fn fidelity_report_summarizes_counts_and_epsilon() {
    let mk = |raw_id: &str| Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![Content::ToolCall {
                id: raw_id.into(),
                name: "f".into(),
                args: json!({}),
            }],
        }],
    };
    let s1 = mk("raw_x");
    let s2 = mk("raw_y");
    let s3 = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![Content::ToolCall {
                id: "raw_z".into(),
                name: "g".into(),
                args: json!({}),
            }],
        }],
    };
    let report = fidelity_report(&[s1, s2, s3], &RawJsonConsumer);
    // (0,1) same nf, raw ids differ → sep violation; (0,2) and (1,2) differ in name,
    // distinct nfs and distinct responses → no blind. Epsilon is 1.0 on the same-nf subset.
    assert_eq!(report.sep_violations, 1);
    assert_eq!(report.blind_violations, 0);
    assert_eq!(report.epsilon, 1.0);
    assert_eq!(report.sep_phi_diffs.len(), 1);
    // Same-nf pair → Φ diffs all zero.
    let (_, _, diff) = report.sep_phi_diffs[0];
    assert_eq!(diff.d_g, 0);
    assert_eq!(diff.d_sigma, 0);
    assert_eq!(diff.d_r, 0);
}

#[test]
fn fidelity_report_epsilon_zero_when_consumer_respects_nf() {
    let s1 = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![text("a")] }],
    };
    let s2 = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![text("a")] }],
    };
    let report = fidelity_report(&[s1, s2], &TextOnlyConsumer);
    assert_eq!(report.epsilon, 0.0);
    assert_eq!(report.sep_violations, 0);
}
