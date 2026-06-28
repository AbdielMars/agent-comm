//! Turnkey conformance entry (Step2 S2-1/S2-4). Run with `cargo test --features conformance`.
//!
//! This is the one-line gate an external codec author copies: feed your codec + your vectors to
//! `run_conformance`, assert the report passes. Here we run each reference codec against its own
//! wire-shape battery and print the structured report — the conclusion is the report
//! (machine-checkable, reproducible), not prose.
#![cfg(feature = "conformance")]

use agent_comm::codecs::{AnthropicCodec, GeminiCodec, OpenAiCodec, ResponsesCodec};
use agent_comm::conformance::{
    reference_vectors_anthropic, reference_vectors_gemini, reference_vectors_openai,
    reference_vectors_responses, run_conformance, ConformanceCase, ConformanceReport,
};
use agent_comm::ProviderCodec;

fn check(codec: &dyn ProviderCodec, vectors: Vec<ConformanceCase>) -> ConformanceReport {
    let report = run_conformance(codec, &vectors);
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    report
}

#[test]
fn all_four_reference_codecs_are_conformant() {
    let reports = [
        check(&AnthropicCodec, reference_vectors_anthropic()),
        check(&OpenAiCodec, reference_vectors_openai()),
        check(&GeminiCodec, reference_vectors_gemini()),
        check(&ResponsesCodec, reference_vectors_responses()),
    ];
    for r in &reports {
        assert!(r.passed(), "{} failed conformance: {:?}", r.provider, r.failures());
    }
}
