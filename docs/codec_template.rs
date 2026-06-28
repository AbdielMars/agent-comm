//! Codec template (Step2 / S2-3). Copy this into `src/codecs/<your_provider>.rs`, register it
//! in `src/codecs/mod.rs` (`mod <p>; pub use <p>::<P>Codec;`) and `codec_for`, fill the TODOs,
//! then write vectors and run `run_conformance`. Pair with docs/CODEC_BUILDER_GUIDE.md.
//!
//! This file is NOT part of the crate build — it is a skeleton to copy.

use crate::codecs::truncated_args_lost; // + cache_directive_lost / placement_collapse_loss as needed
use crate::{CodecError, Content, Conversation, LossObligation, ProviderCodec, Role, Turn};
use serde_json::{json, Value};

pub struct MyCodec;

impl ProviderCodec for MyCodec {
    fn provider_id(&self) -> &'static str {
        "myprovider" // TODO: your stable id
    }

    /// wire → IR. Total or fail-closed; never drop silently (R-3).
    fn up(&self, native: &Value) -> Result<(Conversation, Vec<LossObligation>), CodecError> {
        let mut turns = Vec::new();
        let mut loss: Vec<LossObligation> = Vec::new();

        // TODO 1 — locate the message array in YOUR wire shape (measure it; don't assume).
        let messages = native
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| CodecError::Malformed("myprovider: missing messages[]".into()))?;

        for (idx, msg) in messages.iter().enumerate() {
            // TODO 2 — map role; Role::canon parses common spellings.
            let raw = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let role = Role::canon(raw)
                .ok_or_else(|| CodecError::Malformed(format!("myprovider: bad role '{raw}'")))?;

            // TODO 3 — map each wire element onto a kernel-K generator:
            //   text            -> Content::Text { text, cache_control }
            //   tool call        -> Content::ToolCall { id, name, args }   (args must be an object;
            //                       a non-object / unparseable string -> truncated_args_lost(..))
            //   tool result      -> Content::ToolResult { ref_id, payload, cache_control }
            //   reasoning        -> Content::Thinking { text, sig, placement }
            //   image/video/...  -> Content::Media / Content::Video
            // For anything your wire carries but the IR cannot, push a typed loss into `loss`.
            let _ = (&mut loss, idx, &mut turns, role); // remove once filled

            // turns.push(Turn { role, content });
        }

        Ok((Conversation { turns }, loss))
    }

    /// IR → wire. The mirror: anything YOUR wire cannot express must be a typed loss too.
    fn down(&self, conv: &Conversation) -> Result<(Value, Vec<LossObligation>), CodecError> {
        let mut loss: Vec<LossObligation> = Vec::new();
        let mut messages = Vec::new();

        for (idx, turn) in conv.turns.iter().enumerate() {
            for c in &turn.content {
                match c {
                    Content::Text { text, cache_control } => {
                        let _ = (text, cache_control, idx); // TODO 4 — emit text; cache_control -> cache_directive_lost if unsupported
                    }
                    Content::ToolCall { .. } => { /* TODO */ }
                    Content::ToolResult { .. } => { /* TODO */ }
                    Content::Thinking { .. } => { /* TODO: placement_collapse_loss if you collapse placement */ }
                    Content::Media { .. } => { /* TODO or LossObligation::new(.., "media", ..) */ }
                    Content::Video { .. } => loss.push(LossObligation::new(
                        "myprovider", "video", idx, false, "no video channel",
                    )),
                }
            }
            let _ = json!({}); // build your wire message here
        }

        Ok((json!({ "messages": messages }), loss))
    }
}

// Then, in tests:
//   let report = run_conformance(&MyCodec, &my_vectors_in_my_wire_shape());
//   assert!(report.passed());
