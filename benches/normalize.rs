use criterion::{black_box, criterion_group, criterion_main, Criterion};
use serde_json::json;

// ============================================================================
// Data generators (4 scales: 1 / 10 / 100 / 1000 turns)
// ============================================================================

fn make_turn_text(text: &str) -> serde_json::Value {
    json!({"role": "user", "content": [{"type": "text", "text": text}]})
}

fn make_turn_text_and_tool_call(id: &str, name: &str, args: serde_json::Value) -> serde_json::Value {
    json!({
        "role": "assistant",
        "content": [
            {"type": "text", "text": "some reasoning"},
            {"type": "tool_use", "id": id, "name": name, "input": args}
        ]
    })
}

fn make_turn_tool_result(ref_id: &str, payload: &str) -> serde_json::Value {
    json!({
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": ref_id, "content": payload}]
    })
}

fn deep_nested_args(depth: u32) -> serde_json::Value {
    if depth == 0 {
        json!({"a": 1, "b": 2, "c": 3})
    } else {
        json!({"z": 9, "inner": deep_nested_args(depth - 1), "a": 1})
    }
}

/// Generate an anthropic-wire conversation with `n` turns.
/// Pattern: user → assistant(tool_call) → user(tool_result) → ...
/// Tool calls have deep-nested args (depth=4) to stress R2 canon_value.
fn gen_conversation(n: usize) -> serde_json::Value {
    let mut messages = Vec::new();
    for i in 0..n {
        messages.push(make_turn_text(&format!("message {i}")));
        let id = format!("tool_{i}");
        messages.push(make_turn_text_and_tool_call(
            &id,
            "search",
            deep_nested_args(4),
        ));
        messages.push(make_turn_tool_result(&id, &format!("result {i}")));
    }
    json!({"messages": messages})
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_normalize(c: &mut Criterion) {
    use agent_comm::codecs::AnthropicCodec;
    use agent_comm::ProviderCodec;

    for &scale in &[1usize, 10, 100, 1000] {
        let native = gen_conversation(scale);
        let codec = AnthropicCodec;
        let (conv, _) = codec.up(&native).unwrap();

        // measure Conversation::normalize (full pipeline: R5→R1→R6→R3→R2)
        c.bench_function(&format!("normalize/{scale}turns"), |b| {
            b.iter(|| {
                let _nf = black_box(&conv).normalize();
            })
        });

        // measure encode (calls normalize internally + down)
        c.bench_function(&format!("encode/{scale}turns"), |b| {
            b.iter(|| {
                let nf = black_box(&conv).normalize();
                let _res = codec.down(&nf);
            })
        });

        // measure decode (up + normalize)
        c.bench_function(&format!("decode/{scale}turns"), |b| {
            b.iter(|| {
                let _result = codec.up(black_box(&native));
            })
        });
    }
}

criterion_group!(benches, bench_normalize);
criterion_main!(benches);
