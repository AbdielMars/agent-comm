//! OpenAI codec — flat `messages[]`. Assistant carries `tool_calls`; `tool_result` lives
//! in standalone `role: "tool"` messages. `content` and `tool_calls` are separate fields
//! → intra-turn order between text and tool calls is not expressible (R-3 loss on flatten).

use crate::codecs::{intra_turn_order_lost, parse_data_uri, payload_text_with_loss};
use crate::{CodecError, Content, Conversation, LossObligation, Placement, ProviderCodec, Role, Turn};
use serde_json::{json, Value};

pub struct OpenAiCodec;

impl ProviderCodec for OpenAiCodec {
    fn provider_id(&self) -> &'static str {
        "openai"
    }

    fn up(&self, native: &Value) -> Result<(Conversation, Vec<LossObligation>), CodecError> {
        let messages = native
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| CodecError::Malformed("openai: missing messages[]".into()))?;

        let mut turns = Vec::new();
        let mut loss: Vec<LossObligation> = Vec::new();
        for (idx, msg) in messages.iter().enumerate() {
            let raw = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let role = Role::canon(raw).ok_or_else(|| {
                CodecError::Malformed(format!("openai: unknown role '{raw}'"))
            })?;

            // role:tool → standalone tool_result (openai places tool outputs in their own message).
            if role == Role::Tool {
                let ref_id = msg
                    .get("tool_call_id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let text = match msg.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                turns.push(Turn {
                    role,
                    content: vec![Content::ToolResult {
                        ref_id,
                        payload: vec![Content::Text { text, cache_control: None }],
                        cache_control: None,
                    }],
                });
                continue;
            }

            let mut content = Vec::new();
            // `reasoning_content` (DeepSeek / Kimi non-standard) → Thinking generator
            // (no signature). Placed before main content to preserve order.
            if let Some(rc) = msg.get("reasoning_content").and_then(|r| r.as_str()) {
                if !rc.is_empty() {
                    // openai-compat reasoning_content is an inline string field → InlineString.
                    content.push(Content::Thinking {
                        text: rc.to_string(),
                        sig: None,
                        placement: Placement::InlineString,
                    });
                }
            }
            match msg.get("content") {
                Some(Value::String(s)) => content.push(Content::Text { text: s.clone(), cache_control: None }),
                Some(Value::Array(arr)) => {
                    for part in arr {
                        match part.get("type").and_then(|t| t.as_str()) {
                            Some("text") => {
                                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                    content.push(Content::Text { text: t.to_string(), cache_control: None });
                                }
                            }
                            Some("image_url") => {
                                if let Some(url) =
                                    part.pointer("/image_url/url").and_then(|u| u.as_str())
                                {
                                    if let Some((mime, data)) = parse_data_uri(url) {
                                        content.push(Content::Media { mime, data });
                                    } else {
                                        loss.push(LossObligation::new(
                                            "openai",
                                            "image_url.non_base64",
                                            idx,
                                            false,
                                            "non-base64 image_url not inlinable into IR Media",
                                        ));
                                    }
                                }
                            }
                            other => loss.push(LossObligation::new(
                                "openai",
                                &format!("content_part:{}", other.unwrap_or("?")),
                                idx,
                                false,
                                "openai content part type not representable in IR v0",
                            )),
                        }
                    }
                }
                _ => {}
            }
            // assistant's `tool_calls[]` → ToolCall generators.
            if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tcs {
                    let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                    let func = tc.get("function");
                    let name = func
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = match func.and_then(|f| f.get("arguments")) {
                        // openai encodes args as a JSON string; parse back to Value (lossless).
                        // A truncated/malformed string must NOT silently become {} (#21,
                        // Reasonix `closeTruncatedJSON`) → typed behav.truncated_args loss.
                        Some(Value::String(s)) => match serde_json::from_str(s) {
                            Ok(v) => v,
                            Err(_) => {
                                loss.push(crate::codecs::truncated_args_lost(
                                    "openai",
                                    idx,
                                    "json_parse_failed",
                                    s.len(),
                                ));
                                json!({})
                            }
                        },
                        Some(v) => v.clone(),
                        None => json!({}),
                    };
                    content.push(Content::ToolCall { id, name, args });
                }
            }
            turns.push(Turn { role, content });
        }

        Ok((Conversation { turns }, loss))
    }

    fn down(&self, conv: &Conversation) -> Result<(Value, Vec<LossObligation>), CodecError> {
        let mut loss: Vec<LossObligation> = Vec::new();
        let mut messages = Vec::new();

        for (idx, turn) in conv.turns.iter().enumerate() {
            // openai content/tool_calls are separate fields → cannot express text-after-toolcall
            // interleaving. Record the loss when present.
            if intra_turn_order_lost(&turn.content) {
                loss.push(LossObligation::new(
                    "openai",
                    "intra_turn_order",
                    idx,
                    false,
                    "openai content/tool_calls are separate fields; text-after-tool_call order flattened",
                ));
            }
            let mut text_parts: Vec<&str> = Vec::new();
            let mut media_parts: Vec<(&str, &str)> = Vec::new();
            let mut reasoning_parts: Vec<&str> = Vec::new();
            let mut tool_calls = Vec::new();
            let mut tool_result_msgs = Vec::new();

            for c in &turn.content {
                match c {
                    Content::Text { text, cache_control } => {
                        if cache_control.is_some() {
                            loss.push(crate::codecs::cache_directive_lost("openai", idx));
                        }
                        text_parts.push(text);
                    }
                    Content::ToolCall { id, name, args } => tool_calls.push(json!({
                        "id": id, "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                        }
                    })),
                    // tool_result → standalone role:tool message (openai structure, text-only
                    // channel; media→loss via payload_text_with_loss).
                    Content::ToolResult { ref_id, payload, cache_control } => {
                        if cache_control.is_some() {
                            loss.push(crate::codecs::cache_directive_lost("openai", idx));
                        }
                        tool_result_msgs.push(json!({
                            "role": "tool", "tool_call_id": ref_id,
                            "content": payload_text_with_loss(payload, "openai", idx, &mut loss)
                        }));
                    }
                    // thinking: openai chat carries reasoning only as the non-standard inline
                    // `reasoning_content` (native placement = InlineString). Any other IR
                    // placement collapses → typed behav.placement_collapsed loss (G1 / T3).
                    // There is no signature field → R-3 loss when a signature is present.
                    Content::Thinking { text, sig, placement } => {
                        if let Some(l) = crate::codecs::placement_collapse_loss(
                            "openai",
                            *placement,
                            Placement::InlineString,
                            idx,
                        ) {
                            loss.push(l);
                        }
                        reasoning_parts.push(text);
                        if sig.is_some() {
                            loss.push(LossObligation::new(
                                "openai",
                                "thinking.signature",
                                idx,
                                false,
                                "openai chat completions has no thinking signature field",
                            ));
                        }
                    }
                    Content::Media { mime, data } => media_parts.push((mime, data)),
                    // openai chat has no video input channel → R-3 loss.
                    Content::Video { .. } => loss.push(LossObligation::new(
                        "openai",
                        "video",
                        idx,
                        false,
                        "openai chat completions has no video input channel",
                    )),
                }
            }

            let role = match turn.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };

            let has_msg = !text_parts.is_empty()
                || !tool_calls.is_empty()
                || !media_parts.is_empty()
                || !reasoning_parts.is_empty();
            if has_msg {
                let mut msg = json!({ "role": role });
                // content: plain text → string; with media → parts array; all empty → null/"".
                // Only allow `content: null` together with `tool_calls` (openai convention);
                // otherwise (reasoning-only or empty) use "" rather than bare null, to avoid
                // stock-openai rejection of null content without tool_calls.
                if media_parts.is_empty() {
                    if text_parts.is_empty() {
                        msg["content"] = if tool_calls.is_empty() {
                            json!("")
                        } else {
                            Value::Null
                        };
                    } else {
                        msg["content"] = json!(text_parts.join(""));
                    }
                } else {
                    let mut parts = Vec::new();
                    if !text_parts.is_empty() {
                        parts.push(json!({"type": "text", "text": text_parts.join("")}));
                    }
                    for (mime, data) in &media_parts {
                        parts.push(json!({
                            "type": "image_url",
                            "image_url": {"url": format!("data:{mime};base64,{data}")}
                        }));
                    }
                    msg["content"] = json!(parts);
                }
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = json!(tool_calls);
                }
                if !reasoning_parts.is_empty() {
                    // Multiple Thinking blocks joined into one reasoning_content → segment
                    // boundaries lost (R-3 accounted).
                    if reasoning_parts.len() > 1 {
                        loss.push(LossObligation::new(
                            "openai",
                            "thinking.segment_boundaries",
                            idx,
                            false,
                            "multiple thinking blocks flattened into one reasoning_content",
                        ));
                    }
                    msg["reasoning_content"] = json!(reasoning_parts.join("\n"));
                }
                messages.push(msg);
            }
            for trm in tool_result_msgs {
                messages.push(trm);
            }
        }

        Ok((json!({ "messages": messages }), loss))
    }
}
