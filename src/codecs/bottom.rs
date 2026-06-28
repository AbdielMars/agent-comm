//! ⊥ codec — the 0-anchor (n-heptane). The strictly-folding baseline: only kernel-K `Text`
//! survives; every other generator collapses to nothing with a typed [`LossObligation`].
//!
//! Two roles (SPEC v1.6 §7 / v1.7):
//!   1. **coverage completeness** — folding a maximal IR through ⊥ emits a typed loss for every
//!      extension generator, proving none can be silently dropped (the loss image is "loaded").
//!   2. **0-anchor** — ⊥ is the maximal-loss baseline against which ε is normalized (the
//!      n-heptane of the octane scale; the normalization itself is wired in the c-meter step).
//!
//! ⊥ is intentionally NOT round-trip conformant: it is the floor, not a faithful codec.

use crate::{CodecError, Content, Conversation, LossObligation, ProviderCodec, Role, Turn};
use serde_json::{json, Value};

/// The ⊥ (bottom) codec. See module docs.
pub struct BottomCodec;

fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

impl ProviderCodec for BottomCodec {
    fn provider_id(&self) -> &'static str {
        "bottom"
    }

    /// The ⊥ wire is text-only: `{ "messages": [ { "role", "text" } ] }`. Parsing it is total
    /// and lossless (text is kernel K).
    fn up(&self, native: &Value) -> Result<(Conversation, Vec<LossObligation>), CodecError> {
        let msgs = native
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| CodecError::Malformed("bottom: missing messages[]".into()))?;
        let mut turns = Vec::new();
        for m in msgs {
            let raw = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let role = Role::canon(raw)
                .ok_or_else(|| CodecError::Malformed(format!("bottom: bad role '{raw}'")))?;
            let text = m.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string();
            turns.push(Turn {
                role,
                content: vec![Content::Text { text, cache_control: None }],
            });
        }
        Ok((Conversation { turns }, Vec::new()))
    }

    /// Strict fold to the text floor: `Text` survives (its `cache_control` is a typed bill loss);
    /// every other generator is dropped with a typed `bottom.*_dropped` loss. The union of these
    /// over a maximal IR is the coverage-completeness witness.
    fn down(&self, conv: &Conversation) -> Result<(Value, Vec<LossObligation>), CodecError> {
        let mut loss = Vec::new();
        let mut messages = Vec::new();
        for (idx, turn) in conv.turns.iter().enumerate() {
            let mut text = String::new();
            for c in &turn.content {
                match c {
                    Content::Text { text: t, cache_control } => {
                        text.push_str(t);
                        if cache_control.is_some() {
                            loss.push(crate::codecs::cache_directive_lost("bottom", idx));
                        }
                    }
                    Content::ToolCall { .. } => loss.push(LossObligation::new(
                        "bottom",
                        "bottom.tool_call_dropped",
                        idx,
                        false,
                        "⊥ floor expresses only text; tool_call dropped",
                    )),
                    Content::ToolResult { .. } => loss.push(LossObligation::new(
                        "bottom",
                        "bottom.tool_result_dropped",
                        idx,
                        false,
                        "⊥ floor expresses only text; tool_result dropped",
                    )),
                    Content::Thinking { .. } => loss.push(LossObligation::new(
                        "bottom",
                        "bottom.thinking_dropped",
                        idx,
                        false,
                        "⊥ floor expresses only text; thinking dropped",
                    )),
                    Content::Media { .. } => loss.push(LossObligation::new(
                        "bottom",
                        "bottom.media_dropped",
                        idx,
                        false,
                        "⊥ floor expresses only text; media dropped",
                    )),
                    Content::Video { .. } => loss.push(LossObligation::new(
                        "bottom",
                        "bottom.video_dropped",
                        idx,
                        false,
                        "⊥ floor expresses only text; video dropped",
                    )),
                }
            }
            messages.push(json!({"role": role_str(turn.role), "text": text}));
        }
        Ok((json!({ "messages": messages }), loss))
    }
}
