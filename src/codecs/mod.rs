//! Provider codecs over the kernel. Reference implementations; further providers add new
//! modules without changing existing ones (each is a new leg of the colimit).

use crate::{Content, LossObligation, Placement};
use serde_json::{json, Value};

mod anthropic;
mod bottom;
mod gemini;
mod openai;
mod responses;

pub use anthropic::AnthropicCodec;
pub use bottom::BottomCodec;
pub use gemini::GeminiCodec;
pub use openai::OpenAiCodec;
pub use responses::ResponsesCodec;

// ----------------------------------------------------------------------------
// Shared helpers used by codecs
// ----------------------------------------------------------------------------

/// Cache directive lost (SPEC v1.7 / G3, magi v0.3). A vendor that cannot express a block's
/// `cache_control` drops it → typed `bill.cache_directive_lost` with a `{lost_field}` detail.
/// Cor5.2: a `bill.`-prefixed kind is billed against ≈_bill only (no behav loss).
pub(crate) fn cache_directive_lost(provider: &str, turn_index: usize) -> LossObligation {
    let mut detail = std::collections::BTreeMap::new();
    detail.insert("lost_field".to_string(), "cache_control".to_string());
    LossObligation::with_detail(
        provider,
        "bill.cache_directive_lost",
        turn_index,
        false,
        "vendor cannot express block cache_control; cache breakpoint dropped",
        detail,
    )
}

/// Truncated / unstructurable tool-call arguments (需求表 v1.4 §3 / #21, Reasonix
/// counter-example). A tool-call `arguments` payload that cannot be structured — a partial
/// JSON string that fails to parse (openai), or a non-object `input` (anthropic) — must NOT
/// be silently coerced to `{}` (Reasonix `closeTruncatedJSON → "{}"`). The codec emits a
/// typed `behav.truncated_args` loss with a `{signal, raw_len}` detail. Only the raw length
/// is recorded, never the raw content. Cor5.2: behav-prefixed ⟹ billed against ≈_behav only.
pub(crate) fn truncated_args_lost(
    provider: &str,
    turn_index: usize,
    signal: &str,
    raw_len: usize,
) -> LossObligation {
    let mut detail = std::collections::BTreeMap::new();
    detail.insert("signal".to_string(), signal.to_string());
    detail.insert("raw_len".to_string(), raw_len.to_string());
    LossObligation::with_detail(
        provider,
        "behav.truncated_args",
        turn_index,
        false,
        "tool-call arguments could not be structured; recorded as typed loss, not silently {}",
        detail,
    )
}

/// Reasoning placement collapse (SPEC v1.7 / G1, magi v0.3). A codec whose vendor expresses
/// reasoning only in a single `native` placement must collapse any other IR placement to it;
/// since `placement ∈ ≈_𝒪`, that collapse is a typed `behav.placement_collapsed` loss with a
/// `{from_placement, to_placement}` detail. Returns `None` when the placement already matches
/// (no loss). Used by the `down` side of placement-carrying codecs (anthropic / openai).
pub(crate) fn placement_collapse_loss(
    provider: &str,
    ir: Placement,
    native: Placement,
    turn_index: usize,
) -> Option<LossObligation> {
    if ir == native {
        return None; // identity — no collapse
    }
    // magi Q-P severity (SPEC v1.7 §7.3): a collapse is a TrueLoss iff `tool_call_kwargs` is
    // involved — reasoning-as-kwargs drives runtime tool dispatch (Letta), semantic and NOT
    // codec-normalizable. block ↔ inline_string is a FormalNormalize: a formal rewrite (field
    // rename / type lift) a codec resolves → not a loss, no LossObligation. This is a single-
    // layer severity test that strictly strengthens the prior "any collapse = loss" (no true
    // loss dropped; the 2 block↔inline forms stop being false-positive losses).
    let involves_kwargs =
        ir == Placement::ToolCallKwargs || native == Placement::ToolCallKwargs;
    if !involves_kwargs {
        return None; // FormalNormalize
    }
    let mut detail = std::collections::BTreeMap::new();
    detail.insert("from_placement".to_string(), ir.as_str().to_string());
    detail.insert("to_placement".to_string(), native.as_str().to_string());
    Some(LossObligation::with_detail(
        provider,
        "behav.placement_collapsed",
        turn_index,
        false,
        format!(
            "reasoning placement {} collapsed to vendor-native {}",
            ir.as_str(),
            native.as_str()
        ),
        detail,
    ))
}

/// Concatenate text-only items in a `Content` slice (used for `tool_result.payload`
/// projection / `system` projection). Joins with `""` — same convention as R1.
pub(crate) fn collect_text(items: &[Content]) -> String {
    items
        .iter()
        .filter_map(|c| match c {
            Content::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Render `ToolResult.payload` as anthropic `tool_result.content`: plain text → string;
/// contains media → blocks array (text/image). Anthropic natively supports image blocks
/// → lossless.
pub(crate) fn anthropic_tool_result_content(payload: &[Content]) -> Value {
    if !payload.iter().any(|c| matches!(c, Content::Media { .. })) {
        return json!(collect_text(payload));
    }
    let mut blocks = Vec::new();
    for c in payload {
        match c {
            Content::Text { text, .. } => blocks.push(json!({"type": "text", "text": text})),
            Content::Media { mime, data } => blocks.push(json!({
                "type": "image", "source": {"type": "base64", "media_type": mime, "data": data}
            })),
            _ => {} // N1 forbids ToolCall/Thinking/Video in tool_result.payload
        }
    }
    json!(blocks)
}

/// Down-side helper: if `payload` contains non-text (e.g. Media) but the target provider's
/// `tool_result` channel only supports plain text, emit a `LossObligation` (R-3, never silent).
/// Returns the text projection.
pub(crate) fn payload_text_with_loss(
    payload: &[Content],
    provider: &str,
    turn_index: usize,
    loss: &mut Vec<LossObligation>,
) -> String {
    if payload.iter().any(|c| matches!(c, Content::Media { .. })) {
        loss.push(LossObligation::new(
            provider,
            "tool_result.media",
            turn_index,
            false,
            "provider tool_result channel is text-only; media in payload dropped",
        ));
    }
    collect_text(payload)
}

/// Detect intra-turn order loss: openai/responses keep `content` and `tool_calls` /
/// `function_call` as separate fields; the wire cannot express "Text/Media after a
/// ToolCall" in their interleaved order. Anthropic/gemini use a single ordered array,
/// so this check does not apply to them.
pub(crate) fn intra_turn_order_lost(content: &[Content]) -> bool {
    let mut seen_call = false;
    for c in content {
        match c {
            Content::ToolCall { .. } => seen_call = true,
            Content::Text { .. } | Content::Media { .. } | Content::Video { .. } if seen_call => {
                return true
            }
            _ => {}
        }
    }
    false
}

/// Parse a data URI `data:<mime>;base64,<data>` → (mime, data). Used for openai `image_url`.
/// Only decodes when the explicit `;base64` token is present; otherwise the URL is not an
/// inlineable payload.
pub(crate) fn parse_data_uri(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if !meta.split(';').any(|p| p.eq_ignore_ascii_case("base64")) {
        return None;
    }
    let mime = meta.split(';').next().unwrap_or("").to_string();
    Some((mime, data.to_string()))
}
