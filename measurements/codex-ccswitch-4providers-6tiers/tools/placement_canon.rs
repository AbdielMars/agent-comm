//! R7 placement 归一测试 —— 严格对照 SPEC v1.7 §7.3 placement collapse 判据表。
//!
//! 判据表:
//!   from\to        block            tool_call_kwargs   inline_string
//!   block          identity         TrueLoss           FormalNormalize
//!   tool_call_kwargs TrueLoss       identity           TrueLoss
//!   inline_string  FormalNormalize  TrueLoss           identity
//!
//! FormalNormalize(block↔inline_string) ⟹ normal form 相等(可归一)
//! TrueLoss(涉 tool_call_kwargs) ⟹ normal form 不等(保持可区分,behav.placement_collapsed 可测)

use agent_comm::{Content, Conversation, Placement, Role, Turn};

fn thinking_conv(placement: Placement) -> Conversation {
    Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![
                Content::Thinking {
                    text: "same reasoning text".into(),
                    sig: None,
                    placement,
                },
                Content::Text { text: "same answer".into(), cache_control: None },
            ],
        }],
    }
}

// ── FormalNormalize: block ↔ inline_string 必须归一相等 ──

#[test]
fn block_and_inline_are_semantically_equal() {
    let a = thinking_conv(Placement::Block);
    let b = thinking_conv(Placement::InlineString);
    // 这正是 anchor_witness 里 anthropic(block) vs deepseek(inline_string) 的情形:
    // 同一段推理、同一答案、sig 两侧 None,唯一分歧是 placement → §7.3 判 FormalNormalize。
    assert!(
        a.semantic_eq(&b),
        "block 与 inline_string 是 FormalNormalize,normal form 必须相等"
    );
    assert_eq!(a.normalize(), b.normalize());
}

#[test]
fn inline_canonicalizes_to_block() {
    // 归一方向:inline_string → block(back-compat default)。
    let nf = thinking_conv(Placement::InlineString).normalize();
    match &nf.turns[0].content[0] {
        Content::Thinking { placement, .. } => {
            assert_eq!(*placement, Placement::Block, "inline_string 归一后应为 block");
        }
        other => panic!("期望 Thinking,得到 {other:?}"),
    }
}

// ── TrueLoss: 涉 tool_call_kwargs 必须【保持不等】(不被 R7 误伤) ──

#[test]
fn kwargs_vs_block_stays_distinguishable() {
    let block = thinking_conv(Placement::Block);
    let kwargs = thinking_conv(Placement::ToolCallKwargs);
    assert!(
        !block.semantic_eq(&kwargs),
        "block vs tool_call_kwargs 是 TrueLoss,normal form 必须【不等】(否则真损失被误归一)"
    );
}

#[test]
fn kwargs_vs_inline_stays_distinguishable() {
    let inline = thinking_conv(Placement::InlineString);
    let kwargs = thinking_conv(Placement::ToolCallKwargs);
    assert!(
        !inline.semantic_eq(&kwargs),
        "inline_string vs tool_call_kwargs 是 TrueLoss,normal form 必须【不等】"
    );
}

#[test]
fn kwargs_preserved_by_normalize() {
    // tool_call_kwargs 不被 R7 改写(identity)。
    let nf = thinking_conv(Placement::ToolCallKwargs).normalize();
    match &nf.turns[0].content[0] {
        Content::Thinking { placement, .. } => {
            assert_eq!(*placement, Placement::ToolCallKwargs, "kwargs 必须原样保留");
        }
        other => panic!("期望 Thinking,得到 {other:?}"),
    }
}

// ── sig 是独立分歧维度,不被 placement 归一波及 ──

#[test]
fn differing_sig_still_distinguishes() {
    let no_sig = thinking_conv(Placement::Block);
    let with_sig = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![
                Content::Thinking {
                    text: "same reasoning text".into(),
                    sig: Some("vendor_sig_abc".into()),
                    placement: Placement::InlineString,
                },
                Content::Text { text: "same answer".into(), cache_control: None },
            ],
        }],
    };
    // placement 归一了,但 sig 不同 → 仍应不等(signature 是另一个族的分歧)。
    assert!(
        !no_sig.semantic_eq(&with_sig),
        "sig 不同应保持可区分,R7 只归一 placement 不碰 sig"
    );
}
