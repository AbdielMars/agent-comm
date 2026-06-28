//! The conformance suite (SPEC v1.7 §7 / Step2 S2-1). Runs the protocol's gates against a
//! [`ProviderCodec`] over a battery of canned vectors and produces a structured, serializable
//! report — the material basis of "agent-comm compatible": anyone builds a codec, runs the
//! suite, and gets a deterministic PASS/FAIL they can reproduce.
//!
//! This is the *offline, deterministic* suite: it feeds native JSON vectors and runs the gates.
//! Live wire probing (sending real requests) is a later step and not part of this module.
//!
//! Evidence carries schema + numbers only, never raw content (the "hash + token, no raw" line).

use serde::Serialize;
use std::collections::BTreeMap;

/// The outcome of one gate. `Pass`/`Fail`/`NotApplicable` (a gate that does not apply to the
/// vector — e.g. a response-side gate with no response supplied — is `NotApplicable`, distinct
/// from `Pass`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail { reason: String },
    NotApplicable { reason: String },
}

impl Verdict {
    pub fn is_fail(&self) -> bool {
        matches!(self, Verdict::Fail { .. })
    }
    pub fn is_pass(&self) -> bool {
        matches!(self, Verdict::Pass)
    }
}

/// One gate's result within a report: the gate name, its verdict, and evidence (schema +
/// numbers — never raw content).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CheckOutcome {
    pub check: String,
    pub verdict: Verdict,
    pub evidence: BTreeMap<String, String>,
}

impl CheckOutcome {
    pub fn pass(check: &str) -> Self {
        Self { check: check.to_string(), verdict: Verdict::Pass, evidence: BTreeMap::new() }
    }
    pub fn fail(check: &str, reason: impl Into<String>) -> Self {
        Self {
            check: check.to_string(),
            verdict: Verdict::Fail { reason: reason.into() },
            evidence: BTreeMap::new(),
        }
    }
    pub fn na(check: &str, reason: impl Into<String>) -> Self {
        Self {
            check: check.to_string(),
            verdict: Verdict::NotApplicable { reason: reason.into() },
            evidence: BTreeMap::new(),
        }
    }
    /// Attach one evidence key/value (schema + numbers only).
    pub fn with(mut self, key: &str, val: impl Into<String>) -> Self {
        self.evidence.insert(key.to_string(), val.into());
        self
    }
}

/// A full conformance report for one codec over the suite's vectors. Serializable → the
/// conclusion is the JSON (machine-checkable, reproducible), not prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConformanceReport {
    pub provider: String,
    pub outcomes: Vec<CheckOutcome>,
}

impl ConformanceReport {
    /// The codec passes the suite iff no gate failed (NotApplicable does not fail the suite).
    pub fn passed(&self) -> bool {
        self.outcomes.iter().all(|o| !o.verdict.is_fail())
    }
    /// The failing gates (empty ⟹ passed).
    pub fn failures(&self) -> Vec<&CheckOutcome> {
        self.outcomes.iter().filter(|o| o.verdict.is_fail()).collect()
    }
}

// ----------------------------------------------------------------------------
// 件3 · the runner + cases. A case is self-describing; `run_conformance` dispatches.
// Two gate families:
//   - codec-faithfulness (round-trip / orphan / abandoned / truncated-args): PASS = the codec
//     behaved faithfully on the (possibly adversarial) input. The defect is *planted* bait;
//     surfacing it = the codec did its job.
//   - traffic-conformance (model-identity / face-purity): PASS = the response traffic is clean;
//     a finding ⟹ FAIL (the provider/traffic violated conformance).
// A FAIL anywhere ⟹ the (codec, traffic) pair has a conformance issue.
// ----------------------------------------------------------------------------

use crate::check::{
    check_face_purity, check_model_identity, find_abandoned_toolcalls, find_orphan_toolresults,
    Finding,
};
use crate::{check_round_trip, ProviderCodec, ResponseEnvelope};
use serde_json::Value;

/// One conformance scenario. The variant *is* the expectation (an `Orphan` case expects the
/// codec to surface the planted orphan; a `FacePurity` case expects clean usage).
pub enum ConformanceCase {
    /// Kernel vector — `decode∘encode` must equal `normalize` (lossless round-trip).
    RoundTrip { name: String, native: Value },
    /// Native carrying an orphan tool_result — the codec must surface it (#20), not drop it.
    Orphan { name: String, native: Value },
    /// Native carrying a tool_call abandoned mid-conversation — must be surfaced (#19).
    Abandoned { name: String, native: Value },
    /// Native carrying truncated tool-call args — must emit behav.truncated_args (#21), not {}.
    TruncatedArgs { name: String, native: Value },
    /// Response-side: requested model + response envelope — model-identity (#16). Clean = pass.
    ModelIdentity { name: String, requested_model: String, response: ResponseEnvelope },
    /// Response-side: face + response envelope — face-purity (#17). Clean usage = pass.
    FacePurity { name: String, face: String, response: ResponseEnvelope },
}

impl ConformanceCase {
    fn name(&self) -> &str {
        match self {
            ConformanceCase::RoundTrip { name, .. }
            | ConformanceCase::Orphan { name, .. }
            | ConformanceCase::Abandoned { name, .. }
            | ConformanceCase::TruncatedArgs { name, .. }
            | ConformanceCase::ModelIdentity { name, .. }
            | ConformanceCase::FacePurity { name, .. } => name,
        }
    }
}

/// Run the suite: every case through the codec, one [`CheckOutcome`] each.
pub fn run_conformance(codec: &dyn ProviderCodec, cases: &[ConformanceCase]) -> ConformanceReport {
    let outcomes = cases.iter().map(|c| run_case(codec, c)).collect();
    ConformanceReport { provider: codec.provider_id().to_string(), outcomes }
}

fn run_case(codec: &dyn ProviderCodec, case: &ConformanceCase) -> CheckOutcome {
    let name = case.name().to_string();
    match case {
        ConformanceCase::RoundTrip { native, .. } => match check_round_trip(codec, native) {
            Ok(true) => CheckOutcome::pass(&name),
            Ok(false) => CheckOutcome::fail(&name, "round-trip not lossless (nf(reparse) != nf)"),
            Err(e) => CheckOutcome::fail(&name, format!("round-trip errored: {e:?}")),
        },
        ConformanceCase::Orphan { native, .. } => match codec.up(native) {
            Ok((conv, _)) => {
                let n = find_orphan_toolresults(&conv).len();
                if n >= 1 {
                    CheckOutcome::pass(&name).with("orphans_surfaced", n.to_string())
                } else {
                    CheckOutcome::fail(&name, "planted orphan tool_result was not surfaced (dropped)")
                }
            }
            Err(e) => CheckOutcome::fail(&name, format!("up errored: {e:?}")),
        },
        ConformanceCase::Abandoned { native, .. } => match codec.up(native) {
            Ok((conv, _)) => {
                let n = find_abandoned_toolcalls(&conv).len();
                if n >= 1 {
                    CheckOutcome::pass(&name).with("abandoned_surfaced", n.to_string())
                } else {
                    CheckOutcome::fail(&name, "planted abandoned tool_call was not surfaced")
                }
            }
            Err(e) => CheckOutcome::fail(&name, format!("up errored: {e:?}")),
        },
        ConformanceCase::TruncatedArgs { native, .. } => match codec.up(native) {
            Ok((_, loss)) => {
                if loss.iter().any(|l| l.dropped_kind == "behav.truncated_args") {
                    CheckOutcome::pass(&name)
                } else {
                    CheckOutcome::fail(&name, "truncated args silently coerced to {} (no typed loss)")
                }
            }
            Err(e) => CheckOutcome::fail(&name, format!("up errored: {e:?}")),
        },
        ConformanceCase::ModelIdentity { requested_model, response, .. } => {
            match check_model_identity(requested_model, response) {
                None => CheckOutcome::pass(&name),
                Some(Finding::Reroute { echoed_model, .. }) => {
                    CheckOutcome::fail(&name, "model reroute detected (echoed != requested)")
                        .with("echoed_model", echoed_model)
                }
                Some(_) => CheckOutcome::fail(&name, "unexpected finding"),
            }
        }
        ConformanceCase::FacePurity { face, response, .. } => {
            let findings = check_face_purity(face, response);
            if findings.is_empty() {
                CheckOutcome::pass(&name)
            } else {
                let leaked: Vec<String> = findings
                    .iter()
                    .filter_map(|f| match f {
                        Finding::FaceImpurity { leaked_fields, .. } => Some(leaked_fields.join(",")),
                        _ => None,
                    })
                    .collect();
                CheckOutcome::fail(&name, "face impurity detected (foreign usage fields)")
                    .with("leaked_fields", leaked.join(";"))
                    .with("finding_count", findings.len().to_string())
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 件2 · reference vectors (anthropic wire shape). A faithful codec on clean traffic passes
// every one. External codec authors write their own vectors in their wire shape; these are
// the canonical worked example + the basis for the discriminating tests.
// ----------------------------------------------------------------------------

use serde_json::json;

/// The reference conformance battery in anthropic wire shape. A faithful codec + clean traffic
/// ⟹ every outcome PASS. Mix: 1 lossless round-trip, 3 planted codec-faithfulness baits
/// (orphan / abandoned / truncated-args — a faithful codec surfaces all), 2 clean traffic
/// checks (model-identity / face-purity).
pub fn reference_vectors_anthropic() -> Vec<ConformanceCase> {
    vec![
        ConformanceCase::RoundTrip {
            name: "round-trip/clean".into(),
            native: json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "c1", "name": "f", "input": {"q": "x"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "ok"}
                ]}
            ]}),
        },
        ConformanceCase::Orphan {
            name: "orphan/planted".into(),
            native: json!({"messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "ghost", "content": "stray"}
                ]}
            ]}),
        },
        ConformanceCase::Abandoned {
            name: "abandoned/planted".into(),
            native: json!({"messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "c1", "name": "f", "input": {}}
                ]},
                {"role": "user", "content": [{"type": "text", "text": "never mind"}]}
            ]}),
        },
        ConformanceCase::TruncatedArgs {
            name: "truncated-args/planted".into(),
            native: json!({"messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "c1", "name": "f", "input": "{\"path\": \"/foo/ba"}
                ]}
            ]}),
        },
        ConformanceCase::ModelIdentity {
            name: "model-identity/clean".into(),
            requested_model: "claude".into(),
            response: ResponseEnvelope { echoed_model: Some("claude".into()), ..Default::default() },
        },
        ConformanceCase::FacePurity {
            name: "face-purity/clean".into(),
            face: "anthropic".into(),
            response: ResponseEnvelope {
                usage: [("input_tokens".to_string(), 10u64), ("output_tokens".to_string(), 5)]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
        },
    ]
}

// ----------------------------------------------------------------------------
// S2-4 reference vectors for the other three wire shapes. Same 6-gate battery as anthropic;
// a faithful codec on clean traffic passes every one. Shapes mirror each native wire.
// ----------------------------------------------------------------------------

/// openai reference battery (flat messages[]; assistant tool_calls with string args; tool role).
pub fn reference_vectors_openai() -> Vec<ConformanceCase> {
    vec![
        ConformanceCase::RoundTrip {
            name: "round-trip/clean".into(),
            native: json!({"messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"q\":\"x\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ]}),
        },
        ConformanceCase::Orphan {
            name: "orphan/planted".into(),
            native: json!({"messages": [{"role": "tool", "tool_call_id": "ghost", "content": "stray"}]}),
        },
        ConformanceCase::Abandoned {
            name: "abandoned/planted".into(),
            native: json!({"messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]},
                {"role": "user", "content": "never mind"}
            ]}),
        },
        ConformanceCase::TruncatedArgs {
            name: "truncated-args/planted".into(),
            native: json!({"messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"path\":\"/foo/ba"}}
                ]}
            ]}),
        },
        ConformanceCase::ModelIdentity {
            name: "model-identity/clean".into(),
            requested_model: "gpt".into(),
            response: ResponseEnvelope { echoed_model: Some("gpt".into()), ..Default::default() },
        },
        ConformanceCase::FacePurity {
            name: "face-purity/clean".into(),
            face: "openai".into(),
            response: ResponseEnvelope {
                usage: [("prompt_tokens".to_string(), 10u64), ("completion_tokens".to_string(), 5)]
                    .into_iter().collect(),
                ..Default::default()
            },
        },
    ]
}

/// gemini reference battery (contents[] of parts; functionCall/functionResponse).
pub fn reference_vectors_gemini() -> Vec<ConformanceCase> {
    vec![
        ConformanceCase::RoundTrip {
            name: "round-trip/clean".into(),
            native: json!({"contents": [
                {"role": "user", "parts": [{"text": "weather?"}]},
                {"role": "model", "parts": [{"functionCall": {"id": "gc1", "name": "f", "args": {"q": "x"}}}]},
                {"role": "user", "parts": [{"functionResponse": {"id": "gc1", "name": "f", "response": {"content": "ok"}}}]}
            ]}),
        },
        ConformanceCase::Orphan {
            name: "orphan/planted".into(),
            native: json!({"contents": [
                {"role": "user", "parts": [{"functionResponse": {"id": "ghost", "name": "ghost", "response": {"content": "stray"}}}]}
            ]}),
        },
        ConformanceCase::Abandoned {
            name: "abandoned/planted".into(),
            native: json!({"contents": [
                {"role": "model", "parts": [{"functionCall": {"id": "gc1", "name": "f", "args": {}}}]},
                {"role": "user", "parts": [{"text": "never mind"}]}
            ]}),
        },
        ConformanceCase::TruncatedArgs {
            name: "truncated-args/planted".into(),
            native: json!({"contents": [
                {"role": "model", "parts": [{"functionCall": {"id": "gc1", "name": "f", "args": "{\"path\": \"/foo/ba"}}]}
            ]}),
        },
        ConformanceCase::ModelIdentity {
            name: "model-identity/clean".into(),
            requested_model: "gemini-pro".into(),
            response: ResponseEnvelope { echoed_model: Some("gemini-pro".into()), ..Default::default() },
        },
        ConformanceCase::FacePurity {
            name: "face-purity/clean".into(),
            face: "gemini".into(),
            response: ResponseEnvelope {
                usage: [("promptTokenCount".to_string(), 10u64), ("candidatesTokenCount".to_string(), 5)]
                    .into_iter().collect(),
                ..Default::default()
            },
        },
    ]
}

/// responses reference battery (input[] items: message / function_call / function_call_output).
pub fn reference_vectors_responses() -> Vec<ConformanceCase> {
    vec![
        ConformanceCase::RoundTrip {
            name: "round-trip/clean".into(),
            native: json!({"input": [
                {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"q\":\"x\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"}
            ]}),
        },
        ConformanceCase::Orphan {
            name: "orphan/planted".into(),
            native: json!({"input": [{"type": "function_call_output", "call_id": "ghost", "output": "stray"}]}),
        },
        ConformanceCase::Abandoned {
            name: "abandoned/planted".into(),
            native: json!({"input": [
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"role": "user", "content": [{"type": "input_text", "text": "never mind"}]}
            ]}),
        },
        ConformanceCase::TruncatedArgs {
            name: "truncated-args/planted".into(),
            native: json!({"input": [
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"path\":\"/foo/ba"}
            ]}),
        },
        ConformanceCase::ModelIdentity {
            name: "model-identity/clean".into(),
            requested_model: "o1".into(),
            response: ResponseEnvelope { echoed_model: Some("o1".into()), ..Default::default() },
        },
        ConformanceCase::FacePurity {
            name: "face-purity/clean".into(),
            face: "responses".into(),
            response: ResponseEnvelope {
                usage: [("input_tokens".to_string(), 10u64), ("output_tokens".to_string(), 5)]
                    .into_iter().collect(),
                ..Default::default()
            },
        },
    ]
}
