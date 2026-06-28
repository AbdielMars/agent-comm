//! Gemini codec — `{systemInstruction, contents[]}`. Gemini's vendor-accidental structure
//! that the IR canonicalizes away:
//!   1. No `tool` role: `functionResponse` lives in user-role content → R6 folds it into a
//!      dedicated `Tool` turn.
//!   2. `functionCall` / `functionResponse` ids are optional: matched by name+order. On
//!      `up` we synthesize a `ref_id` (mirroring gemini's call-order matching) via a
//!      name-FIFO; R3 then renames everything to `call_0, call_1, ...`.
//!   3. `thoughtSignature` / `generationConfig` are vendor-accidental / request-envelope
//!      → not part of the IR.

use crate::codecs::{collect_text, payload_text_with_loss};
use crate::{
    CodecError, Content, Conversation, LossObligation, Placement, ProviderCodec, Role, Turn,
    VideoSource,
};
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct GeminiCodec;

impl GeminiCodec {
    /// Extract the payload text from a gemini `functionResponse.response`. The down side
    /// wraps payload uniformly as `{content: <text>}`; this peels it back out.
    fn extract_response_text(response: Option<&Value>) -> String {
        match response {
            Some(Value::Object(map)) => match map.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => Value::Object(map.clone()).to_string(),
            },
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        }
    }
}

impl ProviderCodec for GeminiCodec {
    fn provider_id(&self) -> &'static str {
        "gemini"
    }

    fn up(&self, native: &Value) -> Result<(Conversation, Vec<LossObligation>), CodecError> {
        let mut turns = Vec::new();
        let mut loss: Vec<LossObligation> = Vec::new();

        // `systemInstruction { parts: [{text}] }` → System turn.
        if let Some(si) = native
            .get("systemInstruction")
            .or_else(|| native.get("system_instruction"))
        {
            let sys_text = si
                .get("parts")
                .and_then(|p| p.as_array())
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                })
                .unwrap_or_default();
            if !sys_text.is_empty() {
                turns.push(Turn {
                    role: Role::System,
                    content: vec![Content::Text { text: sys_text, cache_control: None }],
                });
            }
        }

        let contents = native
            .get("contents")
            .and_then(|c| c.as_array())
            .ok_or_else(|| CodecError::Malformed("gemini: missing contents[]".into()))?;

        // name-FIFO: synthesize ids for id-less functionCalls; match functionResponse by
        // name in call order.
        let mut pending: HashMap<String, std::collections::VecDeque<String>> = HashMap::new();
        let mut synth: u64 = 0;

        for (idx, content) in contents.iter().enumerate() {
            let role = match content.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
                "model" => Role::Assistant,
                _ => Role::User,
            };
            let parts = content.get("parts").and_then(|p| p.as_array());
            let mut items = Vec::new();
            if let Some(parts) = parts {
                for part in parts {
                    if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                        items.push(Content::Text { text: t.to_string(), cache_control: None });
                    } else if let Some(fc) = part.get("functionCall") {
                        let name = fc
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        let id = match fc.get("id").and_then(|i| i.as_str()) {
                            Some(i) if !i.is_empty() => i.to_string(),
                            _ => {
                                let s = format!("gcall_{synth}");
                                synth += 1;
                                s
                            }
                        };
                        pending.entry(name.clone()).or_default().push_back(id.clone());
                        // Gemini `args` is natively an object. A string = unstructurable args
                        // (truncated body) — typed behav.truncated_args, not silent {} (#21).
                        let args = match fc.get("args") {
                            Some(Value::String(s)) => {
                                loss.push(crate::codecs::truncated_args_lost(
                                    "gemini",
                                    idx,
                                    "non_object_args",
                                    s.len(),
                                ));
                                json!({})
                            }
                            Some(v) => v.clone(),
                            None => json!({}),
                        };
                        items.push(Content::ToolCall { id, name, args });
                    } else if let Some(fr) = part.get("functionResponse") {
                        let name = fr
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        let ref_id = match fr.get("id").and_then(|i| i.as_str()) {
                            Some(i) if !i.is_empty() => i.to_string(),
                            _ => pending
                                .get_mut(&name)
                                .and_then(|q| q.pop_front())
                                .unwrap_or_else(|| {
                                    let s = format!("gorphan_{synth}");
                                    synth += 1;
                                    s
                                }),
                        };
                        let text = Self::extract_response_text(fr.get("response"));
                        items.push(Content::ToolResult {
                            ref_id,
                            payload: vec![Content::Text { text, cache_control: None }],
                            cache_control: None,
                        });
                    } else if let Some(inline) = part
                        .get("inlineData")
                        .or_else(|| part.get("inline_data"))
                    {
                        let mime = inline
                            .get("mimeType")
                            .or_else(|| inline.get("mime_type"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("image/png")
                            .to_string();
                        let data = inline
                            .get("data")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string();
                        // video/* → Video subtype; everything else → Media (image/pdf).
                        if mime.starts_with("video/") {
                            items.push(Content::Video {
                                source: VideoSource::Base64 { data },
                                mime,
                                duration_seconds: None,
                            });
                        } else {
                            items.push(Content::Media { mime, data });
                        }
                    } else if let Some(fd) =
                        part.get("fileData").or_else(|| part.get("file_data"))
                    {
                        // fileData = url reference. video/* → Video{Url}; other non-video
                        // url media is not expressible in IR v0 → record loss.
                        let mime = fd
                            .get("mimeType")
                            .or_else(|| fd.get("mime_type"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("")
                            .to_string();
                        let uri = fd
                            .get("fileUri")
                            .or_else(|| fd.get("file_uri"))
                            .and_then(|u| u.as_str())
                            .unwrap_or("")
                            .to_string();
                        if mime.starts_with("video/") {
                            items.push(Content::Video {
                                source: VideoSource::Url { url: uri },
                                mime,
                                duration_seconds: None,
                            });
                        } else {
                            loss.push(LossObligation::new(
                                "gemini",
                                "filedata.non_video",
                                idx,
                                false,
                                "gemini fileData (non-video url media) not representable in IR v0",
                            ));
                        }
                    } else if part.get("thought").is_some() {
                        // gemini thought part: thoughtSignature is conversation-state; text
                        // (if present) becomes Thinking.
                        let text = part
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();
                        let sig = part
                            .get("thoughtSignature")
                            .and_then(|s| s.as_str())
                            .map(|s| s.to_string());
                        // Gemini reasoning is a dedicated part → placement = Block.
                        items.push(Content::Thinking { text, sig, placement: Placement::Block });
                    } else {
                        loss.push(LossObligation::new(
                            "gemini",
                            "unknown_part",
                            idx,
                            false,
                            "gemini part has no recognized key (text/functionCall/functionResponse/inlineData/thought)",
                        ));
                    }
                }
            }
            turns.push(Turn { role, content: items });
        }

        Ok((Conversation { turns }, loss))
    }

    fn down(&self, conv: &Conversation) -> Result<(Value, Vec<LossObligation>), CodecError> {
        let mut loss: Vec<LossObligation> = Vec::new();

        // functionResponse needs a `name`, but the IR's `ToolResult` only carries `ref_id`
        // → build an id→name map first.
        let mut name_by_id: HashMap<&str, &str> = HashMap::new();
        for turn in &conv.turns {
            for c in &turn.content {
                if let Content::ToolCall { id, name, .. } = c {
                    name_by_id.insert(id.as_str(), name.as_str());
                }
            }
        }

        let mut system_instruction: Option<String> = None;
        // Group into (role, parts), then merge consecutive same-role groups — gemini also
        // alternates user/model.
        let mut grouped: Vec<(&str, Vec<Value>)> = Vec::new();

        for (idx, turn) in conv.turns.iter().enumerate() {
            if turn.role == Role::System {
                let t = collect_text(&turn.content);
                system_instruction = Some(match system_instruction {
                    Some(prev) => format!("{prev}\n\n{t}"),
                    None => t,
                });
                continue;
            }
            let role = match turn.role {
                Role::Assistant => "model",
                _ => "user",
            };
            let mut parts = Vec::new();
            for c in &turn.content {
                match c {
                    Content::Text { text, cache_control } => {
                        if cache_control.is_some() {
                            loss.push(crate::codecs::cache_directive_lost("gemini", idx));
                        }
                        parts.push(json!({"text": text}));
                    }
                    // Emit explicit id (gemini accepts it); pairing shifts from implicit
                    // name-order to explicit id, removing same-name ambiguity.
                    Content::ToolCall { id, name, args } => parts.push(json!({
                        "functionCall": {"id": id, "name": name, "args": args}
                    })),
                    Content::ToolResult { ref_id, payload, cache_control } => {
                        if cache_control.is_some() {
                            loss.push(crate::codecs::cache_directive_lost("gemini", idx));
                        }
                        // ref with no matching tool_call → dangling reference; record (never silent).
                        let name = match name_by_id.get(ref_id.as_str()).copied() {
                            Some(n) => n,
                            None => {
                                loss.push(LossObligation::new(
                                    "gemini",
                                    "tool_result.dangling_ref",
                                    idx,
                                    false,
                                    format!(
                                        "functionResponse ref '{ref_id}' has no matching functionCall name"
                                    ),
                                ));
                                ""
                            }
                        };
                        let text = payload_text_with_loss(payload, "gemini", idx, &mut loss);
                        parts.push(json!({
                            "functionResponse": {"id": ref_id, "name": name, "response": {"content": text}}
                        }));
                    }
                    // gemini request has no separate thinking input channel → entire block
                    // dropped; R-3 records.
                    Content::Thinking { sig, .. } => loss.push(LossObligation::new(
                        "gemini",
                        if sig.is_some() {
                            "thinking+signature"
                        } else {
                            "thinking"
                        },
                        idx,
                        false,
                        "gemini request has no thinking input channel",
                    )),
                    Content::Media { mime, data } => {
                        parts.push(json!({"inlineData": {"mimeType": mime, "data": data}}))
                    }
                    // gemini supports video: base64 → inlineData; url → fileData. Lossless.
                    Content::Video { source, mime, .. } => match source {
                        VideoSource::Base64 { data } => {
                            parts.push(json!({"inlineData": {"mimeType": mime, "data": data}}))
                        }
                        VideoSource::Url { url } => {
                            parts.push(json!({"fileData": {"mimeType": mime, "fileUri": url}}))
                        }
                    },
                }
            }
            if parts.is_empty() {
                continue;
            }
            match grouped.last_mut() {
                Some((prev_role, prev_parts)) if *prev_role == role => prev_parts.extend(parts),
                _ => grouped.push((role, parts)),
            }
        }

        let contents: Vec<Value> = grouped
            .into_iter()
            .map(|(role, parts)| json!({"role": role, "parts": parts}))
            .collect();
        let mut out = json!({ "contents": contents });
        if let Some(s) = system_instruction {
            out["systemInstruction"] = json!({ "parts": [{ "text": s }] });
        }
        Ok((out, loss))
    }
}
