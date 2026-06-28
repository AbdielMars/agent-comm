//! Responses codec — OpenAI Responses API: `{instructions, input[]}`.
//!
//! Wire shape:
//!   - `system` → top-level `instructions` (string)
//!   - message item: `{role, content: [{type: input_text|output_text, text}]}`
//!   - tool call: flat standalone item `{type: function_call, call_id, name, arguments: JSON-string}`
//!   - tool result: flat standalone item `{type: function_call_output, call_id, output}`
//!   - reasoning items / store / include / max_output_tokens — vendor-accidental or
//!     envelope; not part of the IR.

use crate::codecs::{collect_text, intra_turn_order_lost, parse_data_uri, payload_text_with_loss};
use crate::{CodecError, Content, Conversation, LossObligation, ProviderCodec, Role, Turn};
use serde_json::{json, Value};

pub struct ResponsesCodec;

impl ProviderCodec for ResponsesCodec {
    fn provider_id(&self) -> &'static str {
        "responses"
    }

    fn up(&self, native: &Value) -> Result<(Conversation, Vec<LossObligation>), CodecError> {
        let mut turns = Vec::new();
        let mut loss: Vec<LossObligation> = Vec::new();

        if let Some(instr) = native.get("instructions").and_then(|i| i.as_str()) {
            if !instr.is_empty() {
                turns.push(Turn {
                    role: Role::System,
                    content: vec![Content::Text { text: instr.to_string(), cache_control: None }],
                });
            }
        }

        let input = native
            .get("input")
            .and_then(|i| i.as_array())
            .ok_or_else(|| CodecError::Malformed("responses: missing input[]".into()))?;

        for (idx, item) in input.iter().enumerate() {
            match item.get("type").and_then(|t| t.as_str()) {
                Some("function_call") => {
                    let id = item
                        .get("call_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    // Truncated/malformed args string must NOT silently become {} (#21,
                    // Reasonix). Typed behav.truncated_args instead. Same as the openai codec.
                    let args = match item.get("arguments") {
                        Some(Value::String(s)) => match serde_json::from_str(s) {
                            Ok(v) => v,
                            Err(_) => {
                                loss.push(crate::codecs::truncated_args_lost(
                                    "responses",
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
                    turns.push(Turn {
                        role: Role::Assistant,
                        content: vec![Content::ToolCall { id, name, args }],
                    });
                }
                Some("function_call_output") => {
                    let ref_id = item
                        .get("call_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let text = match item.get("output") {
                        Some(Value::String(s)) => s.clone(),
                        Some(other) => other.to_string(),
                        None => String::new(),
                    };
                    turns.push(Turn {
                        role: Role::Tool,
                        content: vec![Content::ToolResult {
                            ref_id,
                            payload: vec![Content::Text { text, cache_control: None }],
                            cache_control: None,
                        }],
                    });
                }
                // reasoning items are server-managed / encrypted: cannot be faithfully
                // lifted into the IR → record loss (R-3, never silent).
                Some("reasoning") => loss.push(LossObligation::new(
                    "responses",
                    "reasoning",
                    idx,
                    false,
                    "responses reasoning item is server-managed/encrypted; not represented in IR",
                )),
                // Unrecognized typed item (not a message): record loss (never silent).
                Some(other) if other != "message" => loss.push(LossObligation::new(
                    "responses",
                    &format!("input_item:{other}"),
                    idx,
                    false,
                    "responses input item type not representable in IR v0",
                )),
                // Message item (no `type`, or `type: "message"`): `{role, content: [parts]}`.
                _ => {
                    let raw = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let role = Role::canon(raw).ok_or_else(|| {
                        CodecError::Malformed(format!("responses: unknown role '{raw}'"))
                    })?;
                    let mut content = Vec::new();
                    match item.get("content") {
                        Some(Value::String(s)) => content.push(Content::Text { text: s.clone(), cache_control: None }),
                        Some(Value::Array(arr)) => {
                            for part in arr {
                                let pt =
                                    part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                if matches!(pt, "input_text" | "output_text" | "text") {
                                    if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                        content.push(Content::Text { text: t.to_string(), cache_control: None });
                                    }
                                } else if pt == "input_image" {
                                    if let Some(url) =
                                        part.get("image_url").and_then(|u| u.as_str())
                                    {
                                        if let Some((mime, data)) = parse_data_uri(url) {
                                            content.push(Content::Media { mime, data });
                                        } else {
                                            loss.push(LossObligation::new(
                                                "responses",
                                                "input_image.non_base64",
                                                idx,
                                                false,
                                                "non-base64 input_image not inlinable into IR Media",
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    turns.push(Turn { role, content });
                }
            }
        }

        Ok((Conversation { turns }, loss))
    }

    fn down(&self, conv: &Conversation) -> Result<(Value, Vec<LossObligation>), CodecError> {
        let mut loss: Vec<LossObligation> = Vec::new();
        let mut instructions: Option<String> = None;
        let mut input = Vec::new();

        for (idx, turn) in conv.turns.iter().enumerate() {
            if turn.role == Role::System {
                let t = collect_text(&turn.content);
                instructions = Some(match instructions {
                    Some(prev) => format!("{prev}\n\n{t}"),
                    None => t,
                });
                continue;
            }
            // responses message content and the flat function_call items are separate →
            // cannot express text-after-toolcall interleaving. Record the loss when present.
            if intra_turn_order_lost(&turn.content) {
                loss.push(LossObligation::new(
                    "responses",
                    "intra_turn_order",
                    idx,
                    false,
                    "responses message content and function_call items are separate; order flattened",
                ));
            }
            let is_assistant = turn.role == Role::Assistant;
            let role_str = match turn.role {
                Role::Assistant => "assistant",
                Role::Tool => "tool",
                _ => "user",
            };
            let ctype = if is_assistant { "output_text" } else { "input_text" };
            let mut content_parts: Vec<Value> = Vec::new(); // text + media (input_image)
            let mut tail = Vec::new(); // function_call / function_call_output flat items (order-preserving)
            for c in &turn.content {
                match c {
                    Content::Text { text, cache_control } => {
                        if cache_control.is_some() {
                            loss.push(crate::codecs::cache_directive_lost("responses", idx));
                        }
                        content_parts.push(json!({"type": ctype, "text": text}))
                    }
                    Content::Media { mime, data } => content_parts.push(json!({
                        "type": "input_image", "image_url": format!("data:{mime};base64,{data}")
                    })),
                    Content::ToolCall { id, name, args } => tail.push(json!({
                        "type": "function_call", "call_id": id, "name": name,
                        "arguments": serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                    })),
                    Content::ToolResult { ref_id, payload, cache_control } => {
                        if cache_control.is_some() {
                            loss.push(crate::codecs::cache_directive_lost("responses", idx));
                        }
                        tail.push(json!({
                            "type": "function_call_output", "call_id": ref_id,
                            "output": payload_text_with_loss(payload, "responses", idx, &mut loss)
                        }));
                    }
                    // responses reasoning input is server-managed / encrypted; the client
                    // cannot faithfully replay it → R-3 accounted.
                    Content::Thinking { sig, .. } => loss.push(LossObligation::new(
                        "responses",
                        if sig.is_some() {
                            "thinking+signature"
                        } else {
                            "thinking"
                        },
                        idx,
                        false,
                        "responses reasoning input is server-managed/encrypted",
                    )),
                    // responses input has no video channel → R-3 loss.
                    Content::Video { .. } => loss.push(LossObligation::new(
                        "responses",
                        "video",
                        idx,
                        false,
                        "responses input has no video channel",
                    )),
                }
            }
            if !content_parts.is_empty() {
                input.push(json!({ "role": role_str, "content": content_parts }));
            }
            for item in tail {
                input.push(item);
            }
        }

        let mut out = json!({ "input": input });
        if let Some(s) = instructions {
            out["instructions"] = json!(s);
        }
        Ok((out, loss))
    }
}
