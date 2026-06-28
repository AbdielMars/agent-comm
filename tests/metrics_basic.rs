//! Basic tests for the metrics module (Φ three-coordinate + ε estimate).

use agent_comm::metrics::{epsilon_estimate, phi_msg, phi_of, PhiCoords, ResponseVector};
use agent_comm::{Content, Conversation, Role, Turn};
use serde_json::json;

fn text(s: &str) -> Content {
    Content::Text { text: s.into(), cache_control: None }
}

fn empty() -> Conversation {
    Conversation::default()
}

#[test]
fn phi_coords_total_sums_three_components() {
    let p = PhiCoords {
        phi_g: 3,
        phi_sigma: 1,
        phi_r: 2,
    };
    assert_eq!(p.total(), 6);
}

#[test]
fn phi_of_empty_conversation_is_zero() {
    let p = phi_of(&empty(), &[]);
    assert_eq!(p, PhiCoords::default());
}

#[test]
fn phi_of_all_text_counts_genesis() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![text("a"), text("b"), text("c")],
        }],
    };
    // R1 merges adjacent text into one block.
    let p = phi_of(&conv, &[]);
    assert_eq!(p.phi_g, 1);
    assert_eq!(p.phi_r, 0);
    assert_eq!(p.phi_sigma, 0);
}

#[test]
fn phi_of_with_tool_call_counts_reification() {
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::Assistant,
                content: vec![Content::ToolCall {
                    id: "a".into(),
                    name: "f".into(),
                    args: json!({}),
                }],
            },
            Turn {
                role: Role::Tool,
                content: vec![Content::ToolResult {
                    cache_control: None,
                    ref_id: "a".into(),
                    payload: vec![text("ok")],
                }],
            },
        ],
    };
    let p = phi_of(&conv, &[]);
    // 1 ToolCall + 1 ToolResult; nested depth 0.
    assert_eq!(p.phi_g, 0);
    assert_eq!(p.phi_r, 2);
}

#[test]
fn phi_of_translate_chain_counts_sigma_changes_only() {
    let conv = empty();
    // 3 hops, 3 distinct codecs → 2 sigma transitions.
    let p = phi_of(&conv, &["anthropic", "openai", "gemini"]);
    assert_eq!(p.phi_sigma, 2);
}

#[test]
fn phi_of_alias_normalization_no_sigma() {
    let conv = empty();
    // claude ≡ anthropic → no sigma transition despite the alias change.
    let p = phi_of(&conv, &["claude", "anthropic", "anthropic"]);
    assert_eq!(p.phi_sigma, 0);
}

#[test]
fn phi_of_same_codec_chain_no_sigma() {
    let conv = empty();
    let p = phi_of(&conv, &["openai", "openai", "openai"]);
    assert_eq!(p.phi_sigma, 0);
}

#[test]
fn phi_msg_ignores_sigma() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![text("hi")],
        }],
    };
    let m = phi_msg(&conv);
    assert_eq!(m, 1);
}

#[test]
fn epsilon_estimate_no_pairs_is_zero() {
    let eps = epsilon_estimate(&[]);
    assert_eq!(eps, 0.0);
}

#[test]
fn epsilon_estimate_zero_when_all_observers_agree() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: vec![text("x")],
        }],
    };
    let mut rv = ResponseVector::new();
    rv.insert("o1", b"r".to_vec(), b"r".to_vec());
    rv.insert("o2", b"r".to_vec(), b"r".to_vec());
    let eps = epsilon_estimate(&[(conv.clone(), conv.clone(), rv)]);
    assert_eq!(eps, 0.0);
}

#[test]
fn epsilon_estimate_half_when_half_disagree() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: vec![text("x")],
        }],
    };
    let mut rv_agree = ResponseVector::new();
    rv_agree.insert("o1", b"r".to_vec(), b"r".to_vec());
    let mut rv_diff = ResponseVector::new();
    rv_diff.insert("o1", b"r".to_vec(), b"s".to_vec());
    let eps = epsilon_estimate(&[
        (conv.clone(), conv.clone(), rv_agree),
        (conv.clone(), conv.clone(), rv_diff),
    ]);
    assert!((eps - 0.5).abs() < 1e-9);
}

#[test]
fn epsilon_estimate_sup_over_observers() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: vec![text("x")],
        }],
    };
    // Observer o1 always agrees; observer o2 always disagrees → sup = 1.0.
    let mut rv = ResponseVector::new();
    rv.insert("o1", b"r".to_vec(), b"r".to_vec());
    rv.insert("o2", b"r".to_vec(), b"s".to_vec());
    let eps = epsilon_estimate(&[(conv.clone(), conv.clone(), rv)]);
    assert_eq!(eps, 1.0);
}

#[test]
fn epsilon_estimate_skips_pairs_with_different_normal_forms() {
    let m1 = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: vec![text("a")],
        }],
    };
    let m2 = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: vec![text("b")],
        }],
    };
    let mut rv = ResponseVector::new();
    rv.insert("o1", b"x".to_vec(), b"y".to_vec()); // would be a violation if counted
    let eps = epsilon_estimate(&[(m1, m2, rv)]);
    // Pair skipped because nf(m1) != nf(m2); no qualifying samples → 0.
    assert_eq!(eps, 0.0);
}

#[test]
fn phi_of_nested_tool_result_counts_depth() {
    // Inner tool_result inside payload (valid per N1 limit D0=2).
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Tool,
            content: vec![Content::ToolResult {
                cache_control: None,
                ref_id: "outer".into(),
                payload: vec![Content::ToolResult {
                    cache_control: None,
                    ref_id: "inner".into(),
                    payload: vec![text("ok")],
                }],
            }],
        }],
    };
    let p = phi_of(&conv, &[]);
    // outer ToolResult = 1 + nested_depth(payload) = 1 + 1 = 2 → phi_r = 2.
    assert_eq!(p.phi_r, 2);
}
