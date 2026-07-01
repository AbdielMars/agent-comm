//! Conformance suite for agent-comm.
//!  - fail-closed on identity (DenyAll ⟹ encode/decode error);
//!  - normal form is idempotent;
//!  - codec round-trip on the kernel is lossless (decode∘encode = normalize; no LossObligation);
//!  - LossObligation is reported (here empty for the kernel, which both codecs carry fully).
//!
//! Gates here are public toys: reference stubs, not a production identity gate.

use agent_comm::codecs::{AnthropicCodec, GeminiCodec, OpenAiCodec, ResponsesCodec};
use agent_comm::{
    apply_envelope, check_round_trip, codec_for, decode, encode, request_fingerprint,
    split_envelope, translate, CacheFreshness, CommError, Content, Conversation, Placement,
    ProviderCodec, Role, Turn, VideoSource,
};

fn text(s: &str) -> Content {
    Content::Text { text: s.into(), cache_control: None }
}

fn media(mime: &str, data: &str) -> Content {
    Content::Media { mime: mime.into(), data: data.into() }
}

fn video_url(url: &str, mime: &str) -> Content {
    Content::Video {
        source: VideoSource::Url { url: url.into() },
        mime: mime.into(),
        duration_seconds: None,
    }
}

fn turn_asst_call(id: &str) -> Turn {
    Turn {
        role: Role::Assistant,
        content: vec![Content::ToolCall { id: id.into(), name: "f".into(), args: json!({}) }],
    }
}
use agent_comm::protocol::{IdentityGate, Principal};
use serde_json::json;

struct AllowAll;
impl IdentityGate for AllowAll {
    fn verify(&self, _w: &Principal) -> bool {
        true
    }
    fn preserved(&self, _w: &Principal) -> bool {
        true
    }
}
struct DenyAll;
impl IdentityGate for DenyAll {
    fn verify(&self, _w: &Principal) -> bool {
        false
    }
    fn preserved(&self, _w: &Principal) -> bool {
        false
    }
}
fn who() -> Principal {
    Principal::new("actor-A")
}

fn sample() -> Conversation {
    Conversation {
        turns: vec![
            Turn { role: Role::User, content: vec![Content::Text { text: "hello".into(), cache_control: None }] },
            Turn {
                role: Role::Assistant,
                content: vec![Content::ToolCall {
                    id: "x1".into(),
                    name: "search".into(),
                    args: json!({"q": "rust"}),
                }],
            },
            Turn {
                role: Role::Tool,
                content: vec![Content::ToolResult {
                    cache_control: None,
                    ref_id: "x1".into(),
                    payload: vec![Content::Text { text: "result".into(), cache_control: None }],
                }],
            },
        ],
    }
}

#[test]
fn fail_closed_encode_decode_deny_all() {
    let conv = sample();
    assert_eq!(encode(&conv, &AnthropicCodec, &who(), &DenyAll).err(), Some(CommError::Unauthorized));
    let (native, _) = encode(&conv, &AnthropicCodec, &who(), &AllowAll).unwrap();
    assert_eq!(decode(&native, &AnthropicCodec, &who(), &DenyAll).err(), Some(CommError::Unauthorized));
}

#[test]
fn normal_form_idempotent() {
    let messy = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: vec![
                Content::Text { text: "a".into(), cache_control: None },
                Content::Text { text: "".into(), cache_control: None },   // R5 drop
                Content::Text { text: "b".into(), cache_control: None },   // R1 merge with "a"
            ],
        }],
    };
    let once = messy.normalize();
    assert_eq!(once, once.normalize());
    assert_eq!(once.turns[0].content.len(), 1); // merged + empty dropped
}

#[test]
fn anthropic_round_trip_lossless_on_kernel() {
    let conv = sample().normalize();
    let (native, dloss) = encode(&conv, &AnthropicCodec, &who(), &AllowAll).unwrap();
    assert!(dloss.is_empty(), "kernel down is lossless");
    let (back, uloss) = decode(&native, &AnthropicCodec, &who(), &AllowAll).unwrap();
    assert!(uloss.is_empty(), "kernel up is lossless");
    assert_eq!(back, conv, "decode∘encode == normalize on the kernel");
}

#[test]
fn openai_round_trip_lossless_on_kernel() {
    let conv = sample().normalize();
    let (native, dloss) = encode(&conv, &OpenAiCodec, &who(), &AllowAll).unwrap();
    assert!(dloss.is_empty());
    let (back, uloss) = decode(&native, &OpenAiCodec, &who(), &AllowAll).unwrap();
    assert!(uloss.is_empty());
    assert_eq!(back, conv, "decode∘encode == normalize on the kernel");
}

// ---- T4: Anthropic codec round-trip on real-shape samples ----

#[test]
fn anthropic_round_trip_plain_text() {
    let native = json!({
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]}
        ]
    });
    assert!(check_round_trip(&AnthropicCodec, &native).unwrap());
}

#[test]
fn anthropic_round_trip_tool_cycle() {
    // assistant emits tool_use → user carries tool_result (anthropic places tool results
    // inside user messages).
    let native = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "toolu_01ABC", "name": "get_weather", "input": {"city": "SF"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01ABC", "content": "sunny 22C"}
            ]}
        ]
    });
    assert!(check_round_trip(&AnthropicCodec, &native).unwrap());
}

#[test]
fn anthropic_round_trip_thinking_and_image() {
    let native = json!({
        "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reasoning here", "signature": "sig_xyz"},
                {"type": "text", "text": "answer"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "ABC"}}
            ]}
        ]
    });
    assert!(check_round_trip(&AnthropicCodec, &native).unwrap());
}

// ---- T5: OpenAI codec round-trip on real-shape samples + cross-vendor equivalence ----

#[test]
fn openai_round_trip_plain_text() {
    let native = json!({
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"}
        ]
    });
    assert!(check_round_trip(&OpenAiCodec, &native).unwrap());
}

#[test]
fn openai_round_trip_tool_cycle() {
    let native = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_xyz", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_xyz", "content": "sunny 22C"}
        ]
    });
    assert!(check_round_trip(&OpenAiCodec, &native).unwrap());
}

#[test]
fn cross_vendor_text_and_tool_call_equivalent() {
    // anthropic and openai should converge on the same IR normal form on the kernel slice.
    let anthropic = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ]}
        ]
    });
    let c_a = AnthropicCodec.up(&anthropic).unwrap().0.normalize();
    let (openai_native, _loss) = OpenAiCodec.down(&c_a).unwrap();
    let c_o = OpenAiCodec.up(&openai_native).unwrap().0.normalize();
    assert_eq!(c_a, c_o, "IR coordinates should agree under anthropic and openai projections");
}

// ---- T6: Gemini codec round-trip + three-way cross-vendor equivalence ----

#[test]
fn gemini_round_trip_plain_text() {
    let native = json!({
        "systemInstruction": {"parts": [{"text": "You are helpful."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi there"}]}
        ]
    });
    assert!(check_round_trip(&GeminiCodec, &native).unwrap());
}

#[test]
fn gemini_round_trip_tool_cycle() {
    let native = json!({
        "contents": [
            {"role": "user", "parts": [{"text": "weather?"}]},
            {"role": "model", "parts": [
                {"text": "checking"},
                {"functionCall": {"id": "gc_1", "name": "get_weather", "args": {"city": "SF"}}}
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"id": "gc_1", "name": "get_weather", "response": {"content": "sunny 22C"}}}
            ]}
        ]
    });
    assert!(check_round_trip(&GeminiCodec, &native).unwrap());
}

#[test]
fn cross_vendor_three_way_text_and_tool_call() {
    // The IR should agree across anthropic, openai, and gemini projections (text + tool_call slice).
    let anthropic = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ]}
        ]
    });
    let c_a = AnthropicCodec.up(&anthropic).unwrap().0.normalize();
    let (oi, _) = OpenAiCodec.down(&c_a).unwrap();
    let c_o = OpenAiCodec.up(&oi).unwrap().0.normalize();
    let (gi, _) = GeminiCodec.down(&c_a).unwrap();
    let c_g = GeminiCodec.up(&gi).unwrap().0.normalize();
    assert_eq!(c_a, c_o);
    assert_eq!(c_a, c_g);
}

// ---- T7: Responses codec round-trip + four-way cross-vendor equivalence ----

#[test]
fn responses_round_trip_plain_text() {
    let native = json!({
        "instructions": "You are helpful.",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hi there"}]}
        ]
    });
    assert!(check_round_trip(&ResponsesCodec, &native).unwrap());
}

#[test]
fn responses_round_trip_tool_cycle() {
    let native = json!({
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {"type": "function_call", "call_id": "call_r1", "name": "get_weather",
             "arguments": "{\"city\":\"SF\"}"},
            {"type": "function_call_output", "call_id": "call_r1", "output": "sunny 22C"}
        ]
    });
    assert!(check_round_trip(&ResponsesCodec, &native).unwrap());
}

#[test]
fn cross_vendor_four_way_tool_cycle() {
    // Full tool cycle (user query → assistant tool_use → tool result) projected through
    // all four codecs; after R6 normalization the IR should agree across projections.
    let anthropic = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny 22C"}
            ]}
        ]
    });
    let responses = json!({
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {"type": "function_call", "call_id": "fc_1", "name": "get_weather",
             "arguments": "{\"city\":\"SF\"}"},
            {"type": "function_call_output", "call_id": "fc_1", "output": "sunny 22C"}
        ]
    });
    let openai = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "oc_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "oc_1", "content": "sunny 22C"}
        ]
    });
    let gemini = json!({
        "contents": [
            {"role": "user", "parts": [{"text": "weather?"}]},
            {"role": "model", "parts": [
                {"functionCall": {"id": "gc_1", "name": "get_weather", "args": {"city": "SF"}}}
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"id": "gc_1", "name": "get_weather", "response": {"content": "sunny 22C"}}}
            ]}
        ]
    });
    let c_a = AnthropicCodec.up(&anthropic).unwrap().0.normalize();
    let c_o = OpenAiCodec.up(&openai).unwrap().0.normalize();
    let c_g = GeminiCodec.up(&gemini).unwrap().0.normalize();
    let c_r = ResponsesCodec.up(&responses).unwrap().0.normalize();
    assert_eq!(c_a, c_o);
    assert_eq!(c_a, c_g);
    assert_eq!(c_a, c_r);
}

// ---- T8: translate + RequestEnvelope + codec_for aliases ----

#[test]
fn translate_anthropic_to_openai_semantic() {
    let anthropic = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ]}
        ]
    });
    let (oi, _loss) = translate("anthropic", "openai", &anthropic).unwrap();
    // Re-lift to verify cross-vendor semantic equivalence at the IR level.
    let c_a = AnthropicCodec.up(&anthropic).unwrap().0.normalize();
    let c_o = OpenAiCodec.up(&oi).unwrap().0.normalize();
    assert_eq!(c_a, c_o);
}

#[test]
fn codec_for_aliases_and_unknown() {
    assert_eq!(codec_for("claude").unwrap().provider_id(), "anthropic");
    assert_eq!(codec_for("anthropic").unwrap().provider_id(), "anthropic");
    assert_eq!(codec_for("chat").unwrap().provider_id(), "openai");
    assert_eq!(codec_for("openai_chat").unwrap().provider_id(), "openai");
    assert_eq!(codec_for("openai").unwrap().provider_id(), "openai");
    assert_eq!(codec_for("gemini").unwrap().provider_id(), "gemini");
    assert_eq!(codec_for("codex").unwrap().provider_id(), "responses");
    assert_eq!(codec_for("codex_chat").unwrap().provider_id(), "responses");
    assert_eq!(codec_for("responses").unwrap().provider_id(), "responses");
    assert!(codec_for("nope").is_none());
}

#[test]
fn envelope_preserves_request_params() {
    let native = json!({
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "temperature": 0.2,
        "tools": [{"name": "search"}],
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "hi"}]
    });
    let (conv, env, _up_loss) = split_envelope("anthropic", &native).unwrap();
    // Conversation keys removed; envelope fields preserved.
    assert!(!env.fields.contains_key("messages"));
    assert!(!env.fields.contains_key("system"));
    assert_eq!(env.fields.get("model").unwrap(), "claude-opus-4-7");
    assert_eq!(env.fields.get("max_tokens").unwrap(), 1024);
    assert_eq!(env.fields.get("temperature").unwrap(), 0.2);
    // Apply round-trip: envelope fields reappear on the rebuilt native.
    let (back, _down_loss) = apply_envelope("anthropic", &conv, &env).unwrap();
    assert_eq!(back.get("model").unwrap(), "claude-opus-4-7");
    assert_eq!(back.get("max_tokens").unwrap(), 1024);
    assert_eq!(back.get("tools").unwrap(), &json!([{"name": "search"}]));
}

#[test]
fn translate_combines_up_and_down_loss() {
    // Up side has no loss for kernel input; down to gemini emits thinking loss (no gemini
    // thinking input channel), so the combined loss list should be non-empty.
    let anthropic = json!({
        "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reasoning"},
                {"type": "text", "text": "answer"}
            ]}
        ]
    });
    let (_gi, loss) = translate("anthropic", "gemini", &anthropic).unwrap();
    assert!(loss.iter().any(|l| l.provider == "gemini" && l.dropped_kind.starts_with("thinking")));
}

// ---- T9 batch 1: normalization / N1 / R-5 / R-4 ----

#[test]
fn normalize_is_idempotent_with_mixed_content() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![
                text(""),
                text("a"),
                text("b"),
                Content::ToolCall { id: "raw1".into(), name: "t".into(), args: json!({}) },
            ],
        }],
    };
    let once = conv.normalize();
    let twice = once.normalize();
    assert_eq!(once, twice);
}

#[test]
fn n1_rejects_toolcall_in_result() {
    let c = Content::ToolResult {
        cache_control: None,
        ref_id: "call_0".into(),
        payload: vec![Content::ToolCall {
            id: "x".into(),
            name: "t".into(),
            args: json!({}),
        }],
    };
    assert!(c.validate(0).is_err());
}

#[test]
fn validate_rejects_dangling_tool_result() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Tool,
            content: vec![Content::ToolResult {
                cache_control: None,
                ref_id: "nope".into(),
                payload: vec![text("x")],
            }],
        }],
    };
    assert!(conv.validate().is_err());
}

#[test]
fn validate_rejects_duplicate_tool_call_id() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![
                Content::ToolCall { id: "dup".into(), name: "f".into(), args: json!({}) },
                Content::ToolCall { id: "dup".into(), name: "g".into(), args: json!({}) },
            ],
        }],
    };
    assert!(conv.validate().is_err());
}

#[test]
fn r5_wire_format_is_stable() {
    // Kernel K wire is frozen — old serialized samples must continue to deserialize.
    let wire = r#"{
        "turns": [
            {"role": "assistant", "content": [
                {"kind": "text", "text": "hi"},
                {"kind": "tool_call", "id": "call_0", "name": "f", "args": {"a": 1}}
            ]},
            {"role": "tool", "content": [
                {"kind": "tool_result", "ref": "call_0", "payload": [{"kind": "text", "text": "ok"}]}
            ]}
        ]
    }"#;
    let conv: Conversation = serde_json::from_str(wire).unwrap();
    assert_eq!(conv.turns.len(), 2);
    match &conv.turns[0].content[1] {
        Content::ToolCall { id, name, .. } => {
            assert_eq!(id, "call_0");
            assert_eq!(name, "f");
        }
        _ => panic!("expected tool_call"),
    }
}

#[test]
fn r5_extension_variants_wire_format_is_stable() {
    let wire = r#"{
        "turns": [
            {"role": "assistant", "content": [
                {"kind": "thinking", "text": "let me think", "sig": "abc123"},
                {"kind": "thinking", "text": "no sig"},
                {"kind": "media", "mime": "image/png", "data": "QkFTRTY0"}
            ]}
        ]
    }"#;
    let conv: Conversation = serde_json::from_str(wire).unwrap();
    assert_eq!(conv.turns[0].content.len(), 3);
}

#[test]
fn r4_codecs_are_independent() {
    let native = json!({"messages": [{"role": "user", "content": "x"}]});
    assert!(check_round_trip(&OpenAiCodec, &native).unwrap());
    let native2 = json!({"messages": [{"role": "user", "content": "x"}]});
    assert!(check_round_trip(&AnthropicCodec, &native2).unwrap());
}

#[test]
fn r4_gemini_independent() {
    let native = json!({"contents": [{"role": "user", "parts": [{"text": "x"}]}]});
    assert!(check_round_trip(&GeminiCodec, &native).unwrap());
}

#[test]
fn r4_responses_independent() {
    let native = json!({"input": [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}]});
    assert!(check_round_trip(&ResponsesCodec, &native).unwrap());
}

// ---- T9 batch 2: R6 / R2 / cross-vendor tool_cycle ----

#[test]
fn r6_splits_tool_results_into_own_tool_turns() {
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::Assistant,
                content: vec![
                    Content::ToolCall { id: "a".into(), name: "f".into(), args: json!({}) },
                    Content::ToolCall { id: "b".into(), name: "g".into(), args: json!({}) },
                ],
            },
            Turn {
                role: Role::User, // anthropic-style: tool_result on user role
                content: vec![
                    Content::ToolResult { ref_id: "a".into(), payload: vec![text("r1")], cache_control: None },
                    Content::ToolResult { ref_id: "b".into(), payload: vec![text("r2")], cache_control: None },
                ],
            },
        ],
    };
    let nf = conv.normalize();
    assert_eq!(nf.turns.len(), 3);
    assert_eq!(nf.turns[1].role, Role::Tool);
    assert_eq!(nf.turns[1].content.len(), 1);
    assert_eq!(nf.turns[2].role, Role::Tool);
    assert!(matches!(nf.turns[1].content[0], Content::ToolResult { .. }));
}

#[test]
fn normalize_idempotent_with_tool_result_and_args() {
    let conv = Conversation {
        turns: vec![
            Turn {
                role: Role::Assistant,
                content: vec![
                    text("calling"),
                    Content::ToolCall {
                        id: "raw_99".into(),
                        name: "q".into(),
                        args: json!({"z": 1, "a": 2, "m": {"y": 3, "x": 4}}),
                    },
                ],
            },
            Turn {
                role: Role::User,
                content: vec![Content::ToolResult {
                    cache_control: None,
                    ref_id: "raw_99".into(),
                    payload: vec![text("done")],
                }],
            },
        ],
    };
    let once = conv.normalize();
    let twice = once.normalize();
    assert_eq!(once, twice);
}

#[test]
fn r2_sorts_args_keys() {
    let mk = |args: serde_json::Value| Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![Content::ToolCall { id: "x".into(), name: "f".into(), args }],
        }],
    };
    let c1 = mk(json!({"b": 1, "a": 2, "nested": {"d": 4, "c": 3}}));
    let c2 = mk(json!({"a": 2, "nested": {"c": 3, "d": 4}, "b": 1}));
    assert_eq!(c1.normalize(), c2.normalize());
}

#[test]
fn cross_vendor_tool_cycle_two_way_equivalent() {
    let anthropic = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny 22C"}
            ]}
        ]
    });
    let openai = json!({
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_a", "content": "sunny 22C"}
        ]
    });
    let c_a = AnthropicCodec.up(&anthropic).unwrap().0.normalize();
    let c_o = OpenAiCodec.up(&openai).unwrap().0.normalize();
    assert_eq!(c_a, c_o);
}

#[test]
fn down_anthropic_merges_consecutive_tool_results() {
    let native = json!({
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "g", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "r2"}
            ]}
        ]
    });
    assert!(check_round_trip(&AnthropicCodec, &native).unwrap());
    let c = AnthropicCodec.up(&native).unwrap().0.normalize();
    let (out, _loss) = AnthropicCodec.down(&c).unwrap();
    let msgs = out.get("messages").and_then(|m| m.as_array()).unwrap();
    assert_eq!(msgs.len(), 3);
    let last = msgs[2].get("content").and_then(|c| c.as_array()).unwrap();
    assert_eq!(last.len(), 2, "two tool_results should merge into a single user message");
}

#[test]
fn codex_responses_to_chat_via_ir_composition() {
    // Responses → IR → openai(chat): the legacy pairwise responses-to-chat adapter
    // decomposes into `down_openai ∘ up_responses`; re-lifting must agree at the IR.
    let responses = json!({
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"}
        ]
    });
    let c = ResponsesCodec.up(&responses).unwrap().0.normalize();
    let (chat, _l) = OpenAiCodec.down(&c).unwrap();
    let c2 = OpenAiCodec.up(&chat).unwrap().0.normalize();
    assert_eq!(c, c2);
}

// ---- T9 batch 3: extension generators + loss accounting + intra-turn order + video ----

#[test]
fn openai_round_trip_reasoning_and_image() {
    let native = json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZZ"}}
            ]},
            {"role": "assistant", "content": "ok", "reasoning_content": "thinking..."}
        ]
    });
    assert!(check_round_trip(&OpenAiCodec, &native).unwrap());
}

#[test]
fn gemini_round_trip_inline_data() {
    let native = json!({
        "contents": [
            {"role": "user", "parts": [
                {"text": "see"},
                {"inlineData": {"mimeType": "image/webp", "data": "WEBPDATA"}}
            ]}
        ]
    });
    assert!(check_round_trip(&GeminiCodec, &native).unwrap());
}

#[test]
fn thinking_sig_lost_to_openai_is_accounted() {
    let native = json!({
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "deep", "signature": "S"},
            {"type": "text", "text": "hi"}
        ]}]
    });
    let (out, loss) = translate("anthropic", "openai", &native).unwrap();
    // G1 / S2-6 (magi Q-P): anthropic Block → openai InlineString is a FormalNormalize (block↔
    // inline_string), NOT a true loss — no placement_collapsed. Only the wire signature is lost.
    assert_eq!(loss.len(), 1);
    assert!(loss
        .iter()
        .any(|l| l.dropped_kind == "thinking.signature" && !l.recoverable));
    assert!(
        !loss.iter().any(|l| l.dropped_kind == "behav.placement_collapsed"),
        "block↔inline_string is FormalNormalize, not a placement loss"
    );
    let rc = out.pointer("/messages/0/reasoning_content").and_then(|v| v.as_str());
    assert_eq!(rc, Some("deep"));
}

#[test]
fn thinking_lost_to_gemini_and_responses() {
    let native = json!({
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x"},
            {"type": "text", "text": "y"}
        ]}]
    });
    let (_g, loss_g) = translate("anthropic", "gemini", &native).unwrap();
    assert_eq!(loss_g.len(), 1);
    assert_eq!(loss_g[0].provider, "gemini");
    let (_r, loss_r) = translate("anthropic", "responses", &native).unwrap();
    assert_eq!(loss_r.len(), 1);
    assert_eq!(loss_r[0].provider, "responses");
}

#[test]
fn kernel_only_translation_has_no_loss() {
    let native = json!({"messages": [{"role": "user", "content": "hi"}]});
    let (_o, loss) = translate("anthropic", "openai", &native).unwrap();
    assert!(loss.is_empty());
}

#[test]
fn kernel_wire_omits_vendor_accidental_fields() {
    let c = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![Content::Thinking { text: "t".into(), sig: None, placement: Placement::Block }],
        }],
    };
    let wire = serde_json::to_string(&c).unwrap();
    assert!(!wire.contains("\"sig\""));
    assert!(!wire.contains("cache_control"));
    assert!(!wire.contains("signature"));
}

#[test]
fn anthropic_tool_result_with_image_round_trips() {
    let native = json!({
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "shot", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "text", "text": "see"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QQ=="}}
                ]}
            ]}
        ]
    });
    assert!(check_round_trip(&AnthropicCodec, &native).unwrap());
    let (c, _l) = AnthropicCodec.up(&native).unwrap();
    let c = c.normalize();
    let has_media = c.turns.iter().flat_map(|t| &t.content).any(|x| matches!(
        x, Content::ToolResult { payload, .. } if payload.iter().any(|p| matches!(p, Content::Media { .. }))
    ));
    assert!(has_media);
}

#[test]
fn media_in_tool_result_lost_to_openai_is_accounted() {
    let conv = Conversation {
        turns: vec![
            Turn { role: Role::Assistant, content: vec![Content::ToolCall { id: "c0".into(), name: "f".into(), args: json!({}) }] },
            Turn { role: Role::Tool, content: vec![Content::ToolResult { ref_id: "c0".into(), payload: vec![text("img:"), media("image/png", "QQ==")], cache_control: None }] },
        ],
    }.normalize();
    let (_o, loss) = OpenAiCodec.down(&conv).unwrap();
    assert!(loss.iter().any(|l| l.dropped_kind == "tool_result.media"));
}

#[test]
fn anthropic_up_side_unknown_block_is_accounted() {
    let native = json!({"messages": [{"role": "user", "content": [{"type": "video", "url": "x"}]}]});
    let (_c, loss) = AnthropicCodec.up(&native).unwrap();
    assert!(loss.iter().any(|l| l.dropped_kind.starts_with("unknown_block:")));
}

#[test]
fn gemini_dangling_ref_is_accounted() {
    let conv = Conversation {
        turns: vec![Turn { role: Role::Tool, content: vec![Content::ToolResult { ref_id: "nope".into(), payload: vec![text("x")], cache_control: None }] }],
    };
    let (_g, loss) = GeminiCodec.down(&conv).unwrap();
    assert!(loss.iter().any(|l| l.dropped_kind == "tool_result.dangling_ref"));
}

#[test]
fn multiple_thinking_lost_to_openai_is_accounted() {
    let conv = Conversation {
        turns: vec![Turn { role: Role::Assistant, content: vec![
            Content::Thinking { text: "a".into(), sig: None, placement: Placement::Block },
            Content::Thinking { text: "b".into(), sig: None, placement: Placement::Block },
        ] }],
    };
    let (_o, loss) = OpenAiCodec.down(&conv).unwrap();
    assert!(loss.iter().any(|l| l.dropped_kind == "thinking.segment_boundaries"));
}

#[test]
fn normalize_confluent_across_equivalent_representations() {
    let rep1 = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![
                text("hello "), text(""), text("world"),
                Content::ToolCall { id: "raw_X".into(), name: "f".into(), args: json!({"b": 1, "a": 2}) },
            ],
        }],
    };
    let rep2 = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![
                text("hello world"),
                Content::ToolCall { id: "raw_Y".into(), name: "f".into(), args: json!({"a": 2, "b": 1}) },
            ],
        }],
    };
    assert_eq!(rep1.normalize(), rep2.normalize());
    assert_eq!(rep1.normalize(), rep1.normalize().normalize());
}

#[test]
fn anthropic_non_leading_system_position_accounted() {
    let conv = Conversation {
        turns: vec![
            Turn { role: Role::User, content: vec![text("hi")] },
            Turn { role: Role::System, content: vec![text("mid-system")] },
        ],
    };
    let (_a, loss) = AnthropicCodec.down(&conv).unwrap();
    assert!(loss.iter().any(|l| l.dropped_kind == "system.position"));
}

#[test]
fn anthropic_leading_system_no_position_loss() {
    let conv = Conversation {
        turns: vec![
            Turn { role: Role::System, content: vec![text("sys")] },
            Turn { role: Role::User, content: vec![text("hi")] },
        ],
    };
    let (_a, loss) = AnthropicCodec.down(&conv).unwrap();
    assert!(!loss.iter().any(|l| l.dropped_kind == "system.position"));
}

#[test]
fn gemini_down_emits_explicit_call_id() {
    let conv = Conversation {
        turns: vec![
            Turn { role: Role::Assistant, content: vec![Content::ToolCall { id: "c0".into(), name: "f".into(), args: json!({}) }] },
            Turn { role: Role::Tool, content: vec![Content::ToolResult { ref_id: "c0".into(), payload: vec![text("ok")], cache_control: None }] },
        ],
    }.normalize();
    let (g, _l) = GeminiCodec.down(&conv).unwrap();
    let s = g.to_string();
    assert!(s.contains("\"id\""));
}

#[test]
fn gemini_two_same_name_calls_round_trip() {
    let native = json!({
        "contents": [
            {"role": "model", "parts": [
                {"functionCall": {"id": "a", "name": "get", "args": {"k": 1}}},
                {"functionCall": {"id": "b", "name": "get", "args": {"k": 2}}}
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"id": "b", "name": "get", "response": {"content": "2"}}},
                {"functionResponse": {"id": "a", "name": "get", "response": {"content": "1"}}}
            ]}
        ]
    });
    assert!(check_round_trip(&GeminiCodec, &native).unwrap());
}

#[test]
fn r6_drops_empty_turns() {
    let conv = Conversation {
        turns: vec![
            Turn { role: Role::User, content: vec![text("hi")] },
            Turn { role: Role::Assistant, content: vec![text("")] }, // R5 clears → R6 drops
            Turn { role: Role::User, content: vec![] },              // already empty → drops
        ],
    };
    let nf = conv.normalize();
    assert_eq!(nf.turns.len(), 1);
    assert_eq!(nf.turns[0].role, Role::User);
    assert_eq!(nf.turns[0].content, vec![text("hi")]);
}

#[test]
fn r3_ignores_illegal_toolcall_in_payload() {
    let conv = Conversation {
        turns: vec![
            Turn { role: Role::Assistant, content: vec![Content::ToolCall { id: "a".into(), name: "f".into(), args: json!({}) }] },
            Turn { role: Role::Tool, content: vec![Content::ToolResult { ref_id: "a".into(), cache_control: None, payload: vec![
                Content::ToolCall { id: "illegal".into(), name: "x".into(), args: json!({}) },
            ] }] },
        ],
    };
    let nf = conv.normalize();
    match &nf.turns[0].content[0] {
        Content::ToolCall { id, .. } => assert_eq!(id, "call_0"),
        _ => panic!("expected top-level tool_call"),
    }
    assert!(nf.validate().is_err());
}

#[test]
fn openai_down_reasoning_only_uses_empty_string_not_null() {
    let conv = Conversation {
        turns: vec![Turn { role: Role::Assistant, content: vec![Content::Thinking { text: "hmm".into(), sig: None, placement: Placement::Block }] }],
    };
    let (out, _l) = OpenAiCodec.down(&conv).unwrap();
    assert_eq!(out["messages"][0]["content"], json!(""));
    assert_eq!(out["messages"][0]["reasoning_content"], json!("hmm"));
}

#[test]
fn openai_down_null_content_only_with_tool_calls() {
    let conv = Conversation {
        turns: vec![Turn { role: Role::Assistant, content: vec![Content::ToolCall { id: "c0".into(), name: "f".into(), args: json!({}) }] }],
    }.normalize();
    let (out, _l) = OpenAiCodec.down(&conv).unwrap();
    assert!(out["messages"][0]["content"].is_null());
    assert!(out["messages"][0]["tool_calls"].is_array());
}

#[test]
fn anthropic_preserves_intra_turn_order() {
    let native = json!({"messages": [
        {"role": "assistant", "content": [
            {"type": "text", "text": "a"},
            {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
            {"type": "text", "text": "b"}
        ]}
    ]});
    assert!(check_round_trip(&AnthropicCodec, &native).unwrap());
    let (c, _) = AnthropicCodec.up(&native).unwrap();
    let c = c.normalize();
    let kinds: Vec<&str> = c.turns[0].content.iter().map(|x| match x {
        Content::Text { .. } => "text",
        Content::ToolCall { .. } => "call",
        _ => "?",
    }).collect();
    assert_eq!(kinds, vec!["text", "call", "text"]);
}

#[test]
fn gemini_preserves_intra_turn_order() {
    let native = json!({"contents": [
        {"role": "model", "parts": [
            {"text": "a"},
            {"functionCall": {"id": "t1", "name": "f", "args": {}}},
            {"text": "b"}
        ]}
    ]});
    assert!(check_round_trip(&GeminiCodec, &native).unwrap());
    let (c, _) = GeminiCodec.up(&native).unwrap();
    let c = c.normalize();
    let kinds: Vec<&str> = c.turns[0].content.iter().map(|x| match x {
        Content::Text { .. } => "text",
        Content::ToolCall { .. } => "call",
        _ => "?",
    }).collect();
    assert_eq!(kinds, vec!["text", "call", "text"]);
}

#[test]
fn openai_intra_turn_reorder_accounted() {
    let conv = Conversation { turns: vec![Turn { role: Role::Assistant, content: vec![
        Content::ToolCall { id: "c0".into(), name: "f".into(), args: json!({}) },
        Content::Text { text: "after".into(), cache_control: None },
    ] }] };
    let (_o, loss) = OpenAiCodec.down(&conv).unwrap();
    assert!(loss.iter().any(|l| l.dropped_kind == "intra_turn_order"));
}

#[test]
fn openai_canonical_order_no_order_loss() {
    let conv = Conversation { turns: vec![Turn { role: Role::Assistant, content: vec![
        Content::Text { text: "before".into(), cache_control: None },
        Content::ToolCall { id: "c0".into(), name: "f".into(), args: json!({}) },
    ] }] };
    let (_o, loss) = OpenAiCodec.down(&conv).unwrap();
    assert!(!loss.iter().any(|l| l.dropped_kind == "intra_turn_order"));
}

#[test]
fn responses_intra_turn_reorder_accounted() {
    let conv = Conversation { turns: vec![Turn { role: Role::Assistant, content: vec![
        Content::ToolCall { id: "c0".into(), name: "f".into(), args: json!({}) },
        Content::Text { text: "after".into(), cache_control: None },
    ] }] };
    let (_r, loss) = ResponsesCodec.down(&conv).unwrap();
    assert!(loss.iter().any(|l| l.dropped_kind == "intra_turn_order"));
}

#[test]
fn gemini_video_base64_round_trips() {
    let native = json!({"contents": [
        {"role": "user", "parts": [
            {"inlineData": {"mimeType": "video/mp4", "data": "QUJDRA=="}}
        ]}
    ]});
    assert!(check_round_trip(&GeminiCodec, &native).unwrap());
    let (c, _) = GeminiCodec.up(&native).unwrap();
    assert!(matches!(
        c.turns[0].content[0],
        Content::Video { source: VideoSource::Base64 { .. }, .. }
    ));
}

#[test]
fn gemini_video_url_round_trips() {
    let native = json!({"contents": [
        {"role": "user", "parts": [
            {"fileData": {"mimeType": "video/mp4", "fileUri": "gs://bucket/clip.mp4"}}
        ]}
    ]});
    assert!(check_round_trip(&GeminiCodec, &native).unwrap());
    let (c, _) = GeminiCodec.up(&native).unwrap();
    assert!(matches!(
        c.turns[0].content[0],
        Content::Video { source: VideoSource::Url { .. }, .. }
    ));
}

#[test]
fn video_normalize_is_idempotent() {
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::User,
            content: vec![text("watch:"), video_url("gs://x.mp4", "video/mp4")],
        }],
    };
    let once = conv.normalize();
    assert_eq!(once, once.normalize());
}

#[test]
fn video_lost_to_providers_without_capability() {
    let conv = Conversation {
        turns: vec![Turn { role: Role::User, content: vec![video_url("gs://x.mp4", "video/mp4")] }],
    };
    let (_a, la) = AnthropicCodec.down(&conv).unwrap();
    let (_o, lo) = OpenAiCodec.down(&conv).unwrap();
    let (_r, lr) = ResponsesCodec.down(&conv).unwrap();
    assert!(la.iter().any(|l| l.dropped_kind == "video"));
    assert!(lo.iter().any(|l| l.dropped_kind == "video"));
    assert!(lr.iter().any(|l| l.dropped_kind == "video"));
    let (_g, lg) = GeminiCodec.down(&conv).unwrap();
    assert!(!lg.iter().any(|l| l.dropped_kind == "video"));
}

#[test]
fn video_forbidden_in_tool_result_payload() {
    let conv = Conversation {
        turns: vec![
            turn_asst_call("c0"),
            Turn { role: Role::Tool, content: vec![Content::ToolResult {
                cache_control: None,
                ref_id: "c0".into(),
                payload: vec![video_url("gs://x.mp4", "video/mp4")],
            }] },
        ],
    };
    assert!(conv.validate().is_err());
}

#[test]
fn ids_canonicalized_by_normalize() {
    // R3: tool-call ids become call_0.. regardless of input ids.
    let conv = sample().normalize();
    if let Content::ToolCall { id, .. } = &conv.turns[1].content[0] {
        assert_eq!(id, "call_0");
    } else {
        panic!("expected tool call");
    }
    if let Content::ToolResult { ref_id, .. } = &conv.turns[2].content[0] {
        assert_eq!(ref_id, "call_0", "tool_result ref rewritten to canonical id");
    } else {
        panic!("expected tool result");
    }
}

// ============================================================================
// SPEC v1.7 / Phase 1 T1a — placement (G1), request_fingerprint (G8),
// cache_freshness (G9), cache_control typed loss (G3 interim).
// ============================================================================

#[test]
fn g8_request_fingerprint_is_canonical_and_distinguishing() {
    // Same body, different object key order → same fingerprint (canonical).
    let a = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    let b = json!({"messages": [{"role": "user", "content": "hi"}], "model": "m"});
    assert_eq!(
        request_fingerprint(&a),
        request_fingerprint(&b),
        "key-order-only difference must hash the same (canonical)"
    );
    // Different body → different fingerprint.
    let c = json!({"model": "m", "messages": [{"role": "user", "content": "bye"}]});
    assert_ne!(
        request_fingerprint(&a),
        request_fingerprint(&c),
        "different body must hash differently"
    );
}

#[test]
fn g8_split_envelope_carries_fingerprint_and_default_freshness() {
    let native = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    let (_conv, env, _loss) = split_envelope("openai", &native).unwrap();
    assert!(env.request_fingerprint.is_some(), "envelope carries G8 fingerprint");
    assert_eq!(
        env.request_fingerprint.as_deref().unwrap(),
        request_fingerprint(&native),
        "envelope fingerprint == canonical hash of native"
    );
    assert_eq!(env.cache_freshness, CacheFreshness::Unknown, "G9 default = unknown");
}

#[test]
fn g3_anthropic_cache_control_carried_in_ir_round_trip() {
    // T1b: anthropic CARRIES a block cache_control in the IR (≈_bill) — same-vendor round-trip
    // is lossless (no bill.cache_directive_lost), and `down` re-emits the directive.
    let native = json!({
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "long prefix", "cache_control": {"type": "ephemeral"}}]
        }]
    });
    let (conv, loss) = AnthropicCodec.up(&native).unwrap();
    assert!(
        !loss.iter().any(|l| l.dropped_kind == "bill.cache_directive_lost"),
        "anthropic carries cache_control → no bill loss same-vendor"
    );
    match &conv.turns[0].content[0] {
        Content::Text { cache_control, .. } => assert!(cache_control.is_some(), "carried into IR"),
        _ => panic!("expected text block"),
    }
    let (out, _l) = AnthropicCodec.down(&conv.normalize()).unwrap();
    assert!(
        out.pointer("/messages/0/content/0/cache_control").is_some(),
        "anthropic down re-emits cache_control"
    );
}

#[test]
fn g1_placement_survives_serde_round_trip() {
    let c = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![Content::Thinking {
                text: "r".into(),
                sig: None,
                placement: Placement::ToolCallKwargs,
            }],
        }],
    };
    let wire = serde_json::to_string(&c).unwrap();
    assert!(wire.contains("tool_call_kwargs"), "non-default placement serializes");
    let back: Conversation = serde_json::from_str(&wire).unwrap();
    assert_eq!(c, back, "placement survives serde round-trip");
}

#[test]
fn t2_cache_directive_lost_carries_typed_detail_cross_vendor() {
    // SPEC v1.7 / magi v0.3 path A: anthropic carries cache_control, but openai cannot express
    // it → translate fires a bill.cache_directive_lost with a structured `lost_field` detail.
    let native = json!({
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
        }]
    });
    let (_out, loss) = translate("anthropic", "openai", &native).unwrap();
    let cc = loss
        .iter()
        .find(|l| l.dropped_kind == "bill.cache_directive_lost")
        .expect("cache_directive_lost present cross-vendor");
    let detail = cc.detail.as_ref().expect("typed detail present");
    assert_eq!(detail.get("lost_field").map(String::as_str), Some("cache_control"));
}

#[test]
fn t3_placement_collapse_emits_typed_loss_cross_vendor() {
    // IR reasoning placed in tool_call_kwargs (Letta-style) → anthropic (native Block) must
    // collapse it and emit a typed behav.placement_collapsed loss with from/to detail.
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![Content::Thinking {
                text: "r".into(),
                sig: None,
                placement: Placement::ToolCallKwargs,
            }],
        }],
    };
    let (_native, loss) = AnthropicCodec.down(&conv).unwrap();
    let pc = loss
        .iter()
        .find(|l| l.dropped_kind == "behav.placement_collapsed")
        .expect("placement_collapsed present");
    let d = pc.detail.as_ref().expect("typed detail");
    assert_eq!(d.get("from_placement").map(String::as_str), Some("tool_call_kwargs"));
    assert_eq!(d.get("to_placement").map(String::as_str), Some("block"));
}

#[test]
fn t3_matching_placement_no_collapse_loss() {
    // Block placement → anthropic (native Block): the placement matches, no loss.
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![Content::Thinking { text: "r".into(), sig: None, placement: Placement::Block }],
        }],
    };
    let (_native, loss) = AnthropicCodec.down(&conv).unwrap();
    assert!(
        !loss.iter().any(|l| l.dropped_kind == "behav.placement_collapsed"),
        "matching placement must not emit a collapse loss"
    );
}

#[test]
fn t1b_anthropic_cache_control_round_trip_stable() {
    // Two text blocks, only the first cache-marked: R1 must not fuse them (cache boundary),
    // and the anthropic round-trip must be stable (cache_control carried both ways, G3 / T1b).
    let native = json!({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "cacheable prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "tail"}
        ]}]
    });
    assert!(
        check_round_trip(&AnthropicCodec, &native).unwrap(),
        "anthropic cache_control round-trips stably"
    );
}

// ---------------------------------------------------------------------------
// T5a · #20 orphan-toolmsg gate (需求表 v1.4 §3) — structural, no consumer.
// ---------------------------------------------------------------------------

#[test]
fn t5a_closed_conversation_has_no_orphan() {
    // sample() is tool-closed (call x1 → result ref x1) → gate passes (empty).
    use agent_comm::check::find_orphan_toolresults;
    assert!(find_orphan_toolresults(&sample()).is_empty());
}

#[test]
fn t5a_orphan_toolresult_is_detected() {
    // A tool_result whose ref matches no tool_call anywhere → exactly one orphan finding.
    use agent_comm::check::{find_orphan_toolresults, StructuralFinding};
    let conv = Conversation {
        turns: vec![
            turn_asst_call("x1"),
            Turn {
                role: Role::Tool,
                content: vec![Content::ToolResult {
                    cache_control: None,
                    ref_id: "ghost".into(), // <- references no call
                    payload: vec![text("late")],
                }],
            },
        ],
    };
    let found = find_orphan_toolresults(&conv);
    assert_eq!(
        found,
        vec![StructuralFinding::OrphanToolResult { turn_index: 1, ref_id: "ghost".into() }]
    );
}

#[test]
fn t5a_anthropic_up_surfaces_orphan_not_dropped() {
    // Reasonix-style failure = silently drop an orphan tool_result. A faithful codec must
    // surface it into the IR so it can be typed-accounted. Feed anthropic native with an
    // orphan tool_result (no matching tool_use) → up() → the gate still finds it.
    use agent_comm::check::find_orphan_toolresults;
    let native = json!({
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "ghost1", "content": "stray"}
        ]}]
    });
    let (conv, _loss) = AnthropicCodec.up(&native).unwrap();
    let found = find_orphan_toolresults(&conv);
    assert_eq!(found.len(), 1, "orphan tool_result must survive up() (not silently dropped)");
}

// ---------------------------------------------------------------------------
// T5b · #19 interruption-recovery gate (需求表 v1.4 §3) — abandoned tool calls.
// ---------------------------------------------------------------------------

#[test]
fn t5b_trailing_pending_call_is_not_interruption() {
    // A tool call in the final turn is awaiting execution → legitimate, gate passes.
    use agent_comm::check::find_abandoned_toolcalls;
    let conv = Conversation {
        turns: vec![
            Turn { role: Role::User, content: vec![text("go")] },
            turn_asst_call("x1"), // final turn, no result yet → pending, not abandoned
        ],
    };
    assert!(find_abandoned_toolcalls(&conv).is_empty());
}

#[test]
fn t5b_closed_conversation_has_no_abandoned_call() {
    use agent_comm::check::find_abandoned_toolcalls;
    assert!(find_abandoned_toolcalls(&sample()).is_empty());
}

#[test]
fn t5b_abandoned_call_is_detected() {
    // Call x1 in turn 0, then the conversation moves on (turn 1 is unrelated text) with no
    // result for x1 → x1 was abandoned mid-conversation.
    use agent_comm::check::{find_abandoned_toolcalls, StructuralFinding};
    let conv = Conversation {
        turns: vec![
            turn_asst_call("x1"),
            Turn { role: Role::User, content: vec![text("never mind, do this instead")] },
        ],
    };
    assert_eq!(
        find_abandoned_toolcalls(&conv),
        vec![StructuralFinding::AbandonedToolCall { turn_index: 0, call_id: "x1".into() }]
    );
}

#[test]
fn t5b_anthropic_up_surfaces_interruption_not_fabricated() {
    // Reasonix-style failure = fabricate an interruptedToolResult so the call looks answered.
    // A faithful codec leaves the interrupted call honestly unanswered. Feed anthropic native
    // with an assistant tool_use then an unrelated user turn (no tool_result) → up() → the
    // gate surfaces the abandoned call (no fabricated result hiding it).
    use agent_comm::check::find_abandoned_toolcalls;
    let native = json!({
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "x1", "name": "f", "input": {}}
            ]},
            {"role": "user", "content": [{"type": "text", "text": "stop, different question"}]}
        ]
    });
    let (conv, _loss) = AnthropicCodec.up(&native).unwrap();
    assert_eq!(
        find_abandoned_toolcalls(&conv).len(),
        1,
        "interrupted call must be surfaced (not fabricated into an answered call)"
    );
}

// ---------------------------------------------------------------------------
// T5c · #21 truncated-args (需求表 v1.4 §3 / Reasonix counter-example). Truncated tool-call
// arguments must yield a typed behav.truncated_args loss, never silently coerce to {}.
// ---------------------------------------------------------------------------

#[test]
fn t5c_openai_truncated_arguments_is_typed_loss_not_silent() {
    // A truncated JSON arguments string (Reasonix `closeTruncatedJSON` scenario) must be
    // recorded as a typed loss instead of being unwrap_or'd into {}.
    let native = json!({
        "messages": [{"role": "assistant", "tool_calls": [
            {"id": "x1", "type": "function",
             "function": {"name": "f", "arguments": "{\"path\": \"/foo/ba"}}
        ]}]
    });
    let (_conv, loss) = OpenAiCodec.up(&native).unwrap();
    assert!(
        loss.iter().any(|l| l.dropped_kind == "behav.truncated_args" && !l.recoverable),
        "truncated openai arguments must be typed behav.truncated_args, not silent {{}}"
    );
}

#[test]
fn t5c_openai_well_formed_arguments_no_loss() {
    let native = json!({
        "messages": [{"role": "assistant", "tool_calls": [
            {"id": "x1", "type": "function",
             "function": {"name": "f", "arguments": "{\"q\": \"rust\"}"}}
        ]}]
    });
    let (_conv, loss) = OpenAiCodec.up(&native).unwrap();
    assert!(!loss.iter().any(|l| l.dropped_kind == "behav.truncated_args"));
}

#[test]
fn t5c_anthropic_non_object_input_is_typed_loss() {
    // Anthropic input is natively an object; a string input = unstructurable args → typed.
    let native = json!({
        "messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "x1", "name": "f", "input": "{\"path\": \"/foo/ba"}
        ]}]
    });
    let (_conv, loss) = AnthropicCodec.up(&native).unwrap();
    assert!(loss.iter().any(|l| l.dropped_kind == "behav.truncated_args" && !l.recoverable));
}

// ---------------------------------------------------------------------------
// T5d/e/f · 子相2 · ResponseEnvelope + #16 model-identity + #17 face-purity (SPEC v1.7 §2bis).
// ---------------------------------------------------------------------------

fn usage(pairs: &[(&str, u64)]) -> std::collections::BTreeMap<String, u64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

#[test]
fn t5d_response_fingerprint_is_canonical() {
    use agent_comm::response_fingerprint;
    // Same body, different key order → same fingerprint (reuses G8 canon).
    let a = json!({"model": "x", "usage": {"input_tokens": 1, "output_tokens": 2}});
    let b = json!({"usage": {"output_tokens": 2, "input_tokens": 1}, "model": "x"});
    assert_eq!(response_fingerprint(&a), response_fingerprint(&b));
}

#[test]
fn t5e_model_reroute_is_detected() {
    use agent_comm::check::{check_model_identity, Finding};
    use agent_comm::ResponseEnvelope;
    let resp = ResponseEnvelope { echoed_model: Some("v4-flash".into()), ..Default::default() };
    assert_eq!(
        check_model_identity("deepseek-chat", &resp),
        Some(Finding::Reroute {
            requested_model: "deepseek-chat".into(),
            echoed_model: "v4-flash".into(),
        })
    );
}

#[test]
fn t5e_matching_model_no_finding() {
    use agent_comm::check::check_model_identity;
    use agent_comm::ResponseEnvelope;
    let resp = ResponseEnvelope { echoed_model: Some("deepseek-chat".into()), ..Default::default() };
    assert!(check_model_identity("deepseek-chat", &resp).is_none());
}

#[test]
fn t5e_absent_echo_no_finding() {
    // No echoed model to compare → no claim (充分非必要: cannot assert "no reroute").
    use agent_comm::check::check_model_identity;
    use agent_comm::ResponseEnvelope;
    let resp = ResponseEnvelope::default();
    assert!(check_model_identity("deepseek-chat", &resp).is_none());
}

#[test]
fn t5f_pure_face_no_finding() {
    use agent_comm::check::check_face_purity;
    use agent_comm::ResponseEnvelope;
    let resp = ResponseEnvelope {
        usage: usage(&[("input_tokens", 10), ("output_tokens", 5)]),
        ..Default::default()
    };
    assert!(check_face_purity("anthropic", &resp).is_empty());
}

#[test]
fn t5f_foreign_usage_field_raises_behav_and_bill() {
    // OpenAI-style `prompt_tokens` leaking into the Anthropic face: behav main finding +
    // bill side-effect (prompt_tokens IS a usage field of the openai face → Cor5.2 dual-mark).
    use agent_comm::check::{check_face_purity, Finding, FindingScope};
    use agent_comm::ResponseEnvelope;
    let resp = ResponseEnvelope {
        usage: usage(&[("input_tokens", 10), ("prompt_tokens", 9)]),
        ..Default::default()
    };
    let findings = check_face_purity("anthropic", &resp);
    assert_eq!(
        findings,
        vec![
            Finding::FaceImpurity {
                face: "anthropic".into(),
                leaked_fields: vec!["prompt_tokens".into()],
                scope: FindingScope::Behav,
            },
            Finding::ForeignUsageField {
                face: "anthropic".into(),
                polluted_fields: vec!["prompt_tokens".into()],
                scope: FindingScope::Bill,
            },
        ]
    );
}

#[test]
fn t5f_non_usage_leak_is_behav_only() {
    // A leaked field that is NOT a usage field of any face → behav main finding only, no bill.
    use agent_comm::check::{check_face_purity, Finding, FindingScope};
    use agent_comm::ResponseEnvelope;
    let resp = ResponseEnvelope {
        usage: usage(&[("input_tokens", 10), ("weird_vendor_field", 1)]),
        ..Default::default()
    };
    let findings = check_face_purity("anthropic", &resp);
    assert_eq!(
        findings,
        vec![Finding::FaceImpurity {
            face: "anthropic".into(),
            leaked_fields: vec!["weird_vendor_field".into()],
            scope: FindingScope::Behav,
        }]
    );
}

// ---------------------------------------------------------------------------
// S2-1 件1 · conformance-suite report types (BP1: build + serializable).
// ---------------------------------------------------------------------------

#[test]
fn s2_1_jian1_report_serializes() {
    use agent_comm::conformance::{CheckOutcome, ConformanceReport};
    let report = ConformanceReport {
        provider: "anthropic".into(),
        outcomes: vec![
            CheckOutcome::pass("round-trip"),
            CheckOutcome::fail("orphan-toolmsg", "orphan ref 'ghost' silently dropped")
                .with("dropped_refs", "1"),
            CheckOutcome::na("model-identity", "no response envelope supplied"),
        ],
    };
    assert!(!report.passed());
    assert_eq!(report.failures().len(), 1);
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("\"verdict\":\"fail\""));
    assert!(json.contains("\"check\":\"orphan-toolmsg\""));
    // sanity: a pass-only report passes
    let ok = ConformanceReport { provider: "openai".into(), outcomes: vec![CheckOutcome::pass("round-trip")] };
    assert!(ok.passed());
}

// ---------------------------------------------------------------------------
// S2-1 件2/件3 · BP2: a faithful codec passes the reference battery.
// ---------------------------------------------------------------------------

#[test]
fn s2_1_bp2_faithful_anthropic_passes_reference_battery() {
    use agent_comm::conformance::{reference_vectors_anthropic, run_conformance};
    let report = run_conformance(&AnthropicCodec, &reference_vectors_anthropic());
    assert!(report.passed(), "faithful anthropic codec must pass all reference vectors; failures: {:?}", report.failures());
    assert_eq!(report.outcomes.len(), 6);
}

// ---------------------------------------------------------------------------
// S2-1 件5 · BP3: the suite discriminates — a broken codec / bad traffic FAILs.
// ---------------------------------------------------------------------------

// A deliberately broken codec: Reasonix-style, it silently drops tool_results (orphans vanish).
struct DroppingCodec;
impl ProviderCodec for DroppingCodec {
    fn provider_id(&self) -> &'static str {
        "dropping"
    }
    fn up(
        &self,
        native: &serde_json::Value,
    ) -> Result<(Conversation, Vec<agent_comm::LossObligation>), agent_comm::CodecError> {
        let (mut conv, loss) = AnthropicCodec.up(native)?;
        for turn in &mut conv.turns {
            turn.content.retain(|c| !matches!(c, Content::ToolResult { .. }));
        }
        Ok((conv, loss))
    }
    fn down(
        &self,
        conv: &Conversation,
    ) -> Result<(serde_json::Value, Vec<agent_comm::LossObligation>), agent_comm::CodecError> {
        AnthropicCodec.down(conv)
    }
}

#[test]
fn s2_1_bp3_dropping_codec_fails_orphan_check() {
    use agent_comm::conformance::{reference_vectors_anthropic, run_conformance};
    let report = run_conformance(&DroppingCodec, &reference_vectors_anthropic());
    assert!(!report.passed(), "a codec that drops tool_results must not pass the suite");
    let orphan = report.outcomes.iter().find(|o| o.check == "orphan/planted").unwrap();
    assert!(orphan.verdict.is_fail(), "dropping codec must FAIL the orphan check");
}

#[test]
fn s2_1_bp3_bad_traffic_fails_response_gates() {
    use agent_comm::conformance::{run_conformance, ConformanceCase};
    use agent_comm::ResponseEnvelope;
    let cases = vec![
        ConformanceCase::ModelIdentity {
            name: "reroute".into(),
            requested_model: "deepseek-chat".into(),
            response: ResponseEnvelope { echoed_model: Some("v4-flash".into()), ..Default::default() },
        },
        ConformanceCase::FacePurity {
            name: "impure".into(),
            face: "anthropic".into(),
            response: ResponseEnvelope {
                usage: [("prompt_tokens".to_string(), 9u64)].into_iter().collect(),
                ..Default::default()
            },
        },
    ];
    let report = run_conformance(&AnthropicCodec, &cases);
    assert_eq!(report.failures().len(), 2, "both bad-traffic gates must FAIL: {:?}", report.outcomes);
}

// ---------------------------------------------------------------------------
// S2-4 · the other three reference codecs pass their own-shape batteries.
// ---------------------------------------------------------------------------

#[test]
fn s2_4_openai_codec_passes_reference_battery() {
    use agent_comm::conformance::{reference_vectors_openai, run_conformance};
    let r = run_conformance(&OpenAiCodec, &reference_vectors_openai());
    assert!(r.passed(), "openai failures: {:?}", r.failures());
    assert_eq!(r.outcomes.len(), 6);
}

#[test]
fn s2_4_gemini_codec_passes_reference_battery() {
    use agent_comm::conformance::{reference_vectors_gemini, run_conformance};
    let r = run_conformance(&GeminiCodec, &reference_vectors_gemini());
    assert!(r.passed(), "gemini failures: {:?}", r.failures());
    assert_eq!(r.outcomes.len(), 6);
}

#[test]
fn s2_4_responses_codec_passes_reference_battery() {
    use agent_comm::conformance::{reference_vectors_responses, run_conformance};
    let r = run_conformance(&ResponsesCodec, &reference_vectors_responses());
    assert!(r.passed(), "responses failures: {:?}", r.failures());
    assert_eq!(r.outcomes.len(), 6);
}

// ---------------------------------------------------------------------------
// S2-6 · placement collapse severity (magi Q-P / SPEC v1.7 §7.3). FormalNormalize
// (block<->inline_string) is NOT a loss; TrueLoss (involving tool_call_kwargs) is.
// ---------------------------------------------------------------------------

#[test]
fn s2_6_inline_to_block_is_formal_normalize_no_loss() {
    // openai reasoning_content (InlineString) -> anthropic (Block) = FormalNormalize, no loss.
    let native = json!({"messages": [
        {"role": "assistant", "reasoning_content": "hmm", "content": "hi"}
    ]});
    let (_out, loss) = translate("openai", "anthropic", &native).unwrap();
    assert!(
        !loss.iter().any(|l| l.dropped_kind == "behav.placement_collapsed"),
        "InlineString->Block is FormalNormalize, not a placement loss"
    );
}

// ---------------------------------------------------------------------------
// S2-5 ... bottom codec (0-anchor): strict fold to text floor, typed loss per generator.
// Coverage: a maximal IR through down() emits a typed loss for every extension generator.
// ---------------------------------------------------------------------------

#[test]
fn s2_5_bottom_fold_covers_every_extension_generator() {
    use agent_comm::codecs::BottomCodec;
    let conv = Conversation {
        turns: vec![Turn {
            role: Role::Assistant,
            content: vec![
                Content::Text { text: "keep".into(), cache_control: Some(agent_comm::CacheControl { key: None, enabled: true, ttl: None }) },
                Content::ToolCall { id: "c1".into(), name: "f".into(), args: json!({}) },
                Content::ToolResult { ref_id: "c1".into(), payload: vec![text("r")], cache_control: None },
                Content::Thinking { text: "t".into(), sig: None, placement: Placement::Block },
                media("image/png", "AAAA"),
                video_url("u", "video/mp4"),
            ],
        }],
    };
    let (out, loss) = BottomCodec.down(&conv).unwrap();
    // text floor survives
    assert_eq!(out.pointer("/messages/0/text").and_then(|v| v.as_str()), Some("keep"));
    // every extension generator has a typed loss (coverage completeness)
    for kind in [
        "bottom.tool_call_dropped",
        "bottom.tool_result_dropped",
        "bottom.thinking_dropped",
        "bottom.media_dropped",
        "bottom.video_dropped",
        "bill.cache_directive_lost",
    ] {
        assert!(loss.iter().any(|l| l.dropped_kind == kind), "missing typed loss: {kind}");
    }
    // nothing silently dropped: 6 generators beyond bare text -> >= 6 losses
    assert!(loss.len() >= 6);
}
