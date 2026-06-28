//! Anthropic codec — block-structured content (text / tool_use / tool_result / thinking /
//! image), top-level `system` field, no `tool` role (tool_result lives in user-role blocks).

use crate::codecs::{anthropic_tool_result_content, collect_text};
use crate::{
    CacheControl, CodecError, Content, Conversation, LossObligation, Placement, ProviderCodec,
    Role, Turn,
};
use serde_json::{json, Value};

pub struct AnthropicCodec;

impl AnthropicCodec {
    /// Read an anthropic block-level `cache_control` (`{"type": "ephemeral", ...}`) into the
    /// IR type (G3). Anthropic auto-scopes (no client key); presence ⟹ enabled, and the
    /// `ttl` (or `type`) value is preserved as `ttl` for a faithful same-vendor round-trip.
    fn parse_cache_control(block: &Value) -> Option<CacheControl> {
        let cc = block.get("cache_control")?;
        let ttl = cc
            .get("ttl")
            .and_then(|t| t.as_str())
            .or_else(|| cc.get("type").and_then(|t| t.as_str()))
            .map(|s| s.to_string());
        Some(CacheControl { key: None, enabled: true, ttl })
    }

    /// Lower an IR `CacheControl` back to anthropic's `cache_control` object.
    fn cache_control_json(cc: &CacheControl) -> Value {
        json!({ "type": cc.ttl.clone().unwrap_or_else(|| "ephemeral".to_string()) })
    }

    /// Parse anthropic content blocks → IR generators.
    /// Kernel K: text / tool_use / tool_result; extensions: thinking / image(→Media).
    /// Unrepresentable block types → emit `LossObligation` (R-3, up never silent).
    fn parse_blocks(
        blocks: &[Value],
        loss: &mut Vec<LossObligation>,
        turn_index: usize,
    ) -> Vec<Content> {
        let mut out = Vec::new();
        for block in blocks {
            let btype = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            // cache_control is a block-level attribute (≈_bill / G3). Carried in the IR on
            // text / tool_result blocks (parse_cache_control). On other block types the IR has
            // no slot → typed bill loss (never silent, R-3). Cor5.2: bill-only.
            if block.get("cache_control").is_some() && !matches!(btype, "text" | "tool_result") {
                loss.push(crate::codecs::cache_directive_lost("anthropic", turn_index));
            }
            match btype {
                "text" => {
                    if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                        out.push(Content::Text {
                            text: t.to_string(),
                            cache_control: Self::parse_cache_control(block),
                        });
                    }
                }
                "tool_use" => {
                    let id = block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    // Anthropic `input` is natively an object. A string input = args that
                    // could not be structured (truncated body) — must NOT silently coerce to
                    // {} (#21, Reasonix). Absent input = a legitimate no-arg call (not a loss).
                    let args = match block.get("input") {
                        Some(Value::String(s)) => {
                            loss.push(crate::codecs::truncated_args_lost(
                                "anthropic",
                                turn_index,
                                "non_object_input",
                                s.len(),
                            ));
                            json!({})
                        }
                        Some(v) => v.clone(),
                        None => json!({}),
                    };
                    out.push(Content::ToolCall { id, name, args });
                }
                "tool_result" => {
                    let ref_id = block
                        .get("tool_use_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let payload =
                        Self::parse_tool_result_content(block.get("content"), loss, turn_index);
                    out.push(Content::ToolResult {
                        ref_id,
                        payload,
                        cache_control: Self::parse_cache_control(block),
                    });
                }
                "thinking" | "redacted_thinking" => {
                    let text = block
                        .get("thinking")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    let sig = block
                        .get("signature")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    // Anthropic reasoning is a dedicated block → placement = Block.
                    out.push(Content::Thinking { text, sig, placement: Placement::Block });
                }
                "image" => {
                    if let Some(source) = block.get("source") {
                        let mime = source
                            .get("media_type")
                            .and_then(|m| m.as_str())
                            .unwrap_or("image/png")
                            .to_string();
                        let data = source
                            .get("data")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string();
                        out.push(Content::Media { mime, data });
                    }
                }
                // `cache_control` is a vendor-accidental marker: not part of the kernel,
                // not a loss either (no information payload).
                "cache_control" => {}
                other => loss.push(LossObligation::new(
                    "anthropic",
                    &format!("unknown_block:{other}"),
                    turn_index,
                    false,
                    "anthropic content block type not representable in IR v0",
                )),
            }
        }
        out
    }

    /// `tool_result.content` may be string / blocks array / other → normalized to
    /// `payload: Vec<Content>`. `image` in the array becomes `Media`; non-text/image items
    /// → emit `LossObligation` (never silent).
    fn parse_tool_result_content(
        content: Option<&Value>,
        loss: &mut Vec<LossObligation>,
        turn_index: usize,
    ) -> Vec<Content> {
        match content {
            Some(Value::String(s)) => vec![Content::Text { text: s.clone(), cache_control: None }],
            Some(Value::Array(arr)) => {
                let mut out = Vec::new();
                for b in arr {
                    match b.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                                out.push(Content::Text { text: t.to_string(), cache_control: None });
                            }
                        }
                        Some("image") => {
                            if let Some(src) = b.get("source") {
                                let mime = src
                                    .get("media_type")
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("image/png")
                                    .to_string();
                                let data = src
                                    .get("data")
                                    .and_then(|d| d.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                out.push(Content::Media { mime, data });
                            }
                        }
                        _ => {
                            if let Some(s) = b.as_str() {
                                out.push(Content::Text { text: s.to_string(), cache_control: None });
                            } else {
                                loss.push(LossObligation::new(
                                    "anthropic",
                                    "tool_result.unknown_block",
                                    turn_index,
                                    false,
                                    "non text/image block inside tool_result not representable",
                                ));
                            }
                        }
                    }
                }
                out
            }
            Some(other) => vec![Content::Text { text: other.to_string(), cache_control: None }],
            None => Vec::new(),
        }
    }
}

impl ProviderCodec for AnthropicCodec {
    fn provider_id(&self) -> &'static str {
        "anthropic"
    }

    fn up(&self, native: &Value) -> Result<(Conversation, Vec<LossObligation>), CodecError> {
        let mut turns = Vec::new();
        let mut loss: Vec<LossObligation> = Vec::new();

        // Top-level `system` (string or [{text, cache_control}]) → normalized to a System turn.
        if let Some(sys) = native.get("system") {
            let sys_text = match sys {
                Value::String(s) => s.clone(),
                Value::Array(arr) => arr
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            };
            if !sys_text.is_empty() {
                turns.push(Turn {
                    role: Role::System,
                    content: vec![Content::Text { text: sys_text, cache_control: None }],
                });
            }
        }

        let messages = native
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| CodecError::Malformed("anthropic: missing messages[]".into()))?;

        for (idx, msg) in messages.iter().enumerate() {
            let raw = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let role = Role::canon(raw).ok_or_else(|| {
                CodecError::Malformed(format!("anthropic: unknown role '{raw}'"))
            })?;
            let content = match msg.get("content") {
                Some(Value::String(s)) => vec![Content::Text { text: s.clone(), cache_control: None }],
                Some(Value::Array(arr)) => Self::parse_blocks(arr, &mut loss, idx),
                _ => Vec::new(),
            };
            turns.push(Turn { role, content });
        }

        Ok((Conversation { turns }, loss))
    }

    fn down(&self, conv: &Conversation) -> Result<(Value, Vec<LossObligation>), CodecError> {
        let mut loss: Vec<LossObligation> = Vec::new();
        let mut system: Option<String> = None;
        // Anthropic only has a top-level `system` field; mid-conversation system turns
        // cannot be expressed in place. Non-leading System is hoisted → position lost.
        let mut seen_nonsystem = false;
        // Group turns by (role, blocks), then merge consecutive same-role groups —
        // anthropic API requires strict user/assistant alternation. After R6, consecutive
        // tool_result turns become consecutive Tool turns → all map to user; must merge.
        let mut grouped: Vec<(&str, Vec<Value>)> = Vec::new();

        for (idx, turn) in conv.turns.iter().enumerate() {
            if turn.role == Role::System {
                if seen_nonsystem {
                    loss.push(LossObligation::new(
                        "anthropic",
                        "system.position",
                        idx,
                        false,
                        "non-leading system turn hoisted to top-level `system`; position lost",
                    ));
                }
                let t = collect_text(&turn.content);
                system = Some(match system {
                    Some(prev) => format!("{prev}\n{t}"),
                    None => t,
                });
                continue;
            }
            seen_nonsystem = true;
            // Anthropic only has user/assistant; Tool collapses to user (no tool role).
            let role = match turn.role {
                Role::Assistant => "assistant",
                _ => "user",
            };
            let mut blocks = Vec::new();
            for c in &turn.content {
                match c {
                    Content::Text { text, cache_control } => {
                        let mut b = json!({"type": "text", "text": text});
                        if let Some(cc) = cache_control {
                            b["cache_control"] = Self::cache_control_json(cc);
                        }
                        blocks.push(b);
                    }
                    Content::ToolCall { id, name, args } => blocks.push(json!({
                        "type": "tool_use", "id": id, "name": name, "input": args
                    })),
                    Content::ToolResult { ref_id, payload, cache_control } => {
                        let mut b = json!({
                            "type": "tool_result", "tool_use_id": ref_id,
                            "content": anthropic_tool_result_content(payload)
                        });
                        if let Some(cc) = cache_control {
                            b["cache_control"] = Self::cache_control_json(cc);
                        }
                        blocks.push(b);
                    }
                    // Anthropic expresses reasoning only as a `thinking` block (native
                    // placement = Block). Any other IR placement collapses to Block → typed
                    // behav.placement_collapsed loss (G1 / T3). Signature carried as-is.
                    Content::Thinking { text, sig, placement } => {
                        if let Some(l) = crate::codecs::placement_collapse_loss(
                            "anthropic",
                            *placement,
                            Placement::Block,
                            idx,
                        ) {
                            loss.push(l);
                        }
                        let mut b = json!({"type": "thinking", "thinking": text});
                        if let Some(s) = sig {
                            b["signature"] = json!(s);
                        }
                        blocks.push(b);
                    }
                    Content::Media { mime, data } => blocks.push(json!({
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": data}
                    })),
                    // Anthropic request has no video input channel → emit loss (R-3, never silent).
                    Content::Video { .. } => loss.push(LossObligation::new(
                        "anthropic",
                        "video",
                        idx,
                        false,
                        "anthropic request has no video input channel",
                    )),
                }
            }
            if blocks.is_empty() {
                continue;
            }
            match grouped.last_mut() {
                Some((prev_role, prev_blocks)) if *prev_role == role => prev_blocks.extend(blocks),
                _ => grouped.push((role, blocks)),
            }
        }

        let messages: Vec<Value> = grouped
            .into_iter()
            .map(|(role, blocks)| json!({"role": role, "content": blocks}))
            .collect();
        let mut out = json!({ "messages": messages });
        if let Some(s) = system {
            out["system"] = json!(s);
        }
        Ok((out, loss))
    }
}
