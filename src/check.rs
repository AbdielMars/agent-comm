//! Fidelity checks over the IR.
//!
//! - `find_blind_violation` — searches for sample pairs whose normal forms differ but the
//!   consumer's responses agree (the IR drew a distinction the consumer doesn't observe).
//! - `find_sep_violation` — searches for sample pairs whose normal forms are equal but
//!   the consumer's responses differ (the IR fails to draw a distinction the consumer
//!   does observe).
//! - `fidelity_report` — produces a structured report (ε estimate, violation counts,
//!   the Φ coordinate that "should have" distinguished the pair).
//!
//! The consumer is modeled by the `ConsumerFn` trait: a deterministic function from
//! conversation to response. The checks iterate over all unordered sample pairs.

use crate::metrics::{epsilon_estimate, phi_of, PhiCoords, ResponseVector};
use crate::{Content, Conversation, ResponseEnvelope};

/// A downstream consumer that produces a response for a conversation. Deterministic for
/// the purpose of these checks.
pub trait ConsumerFn {
    fn observe(&self, conv: &Conversation) -> Vec<u8>;
}

/// Blanket impl: any deterministic closure is a `ConsumerFn`.
impl<F> ConsumerFn for F
where
    F: Fn(&Conversation) -> Vec<u8>,
{
    fn observe(&self, conv: &Conversation) -> Vec<u8> {
        (self)(conv)
    }
}

/// A faithfulness violation: a sample pair plus the kind it violates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// Two conversations whose normal forms differ but whose consumer responses agree.
    /// The IR is "too sharp" for this consumer on this pair.
    Blind {
        i: usize,
        j: usize,
        response: Vec<u8>,
    },
    /// Two conversations whose normal forms are equal but whose consumer responses
    /// differ. The IR is "too coarse" — it failed to record a distinction the consumer
    /// does observe.
    Sep {
        i: usize,
        j: usize,
        response_i: Vec<u8>,
        response_j: Vec<u8>,
    },
}

/// Find the first sample pair (i, j) where `nf(samples[i]) != nf(samples[j])` but
/// `observe(samples[i]) == observe(samples[j])`. Returns `None` if no such pair exists.
pub fn find_blind_violation(
    samples: &[Conversation],
    consumer: &dyn ConsumerFn,
) -> Option<Violation> {
    let nfs: Vec<Conversation> = samples.iter().map(|c| c.normalize()).collect();
    let responses: Vec<Vec<u8>> = samples.iter().map(|c| consumer.observe(c)).collect();
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            if nfs[i] != nfs[j] && responses[i] == responses[j] {
                return Some(Violation::Blind {
                    i,
                    j,
                    response: responses[i].clone(),
                });
            }
        }
    }
    None
}

/// Find the first sample pair (i, j) where `nf(samples[i]) == nf(samples[j])` but
/// `observe(samples[i]) != observe(samples[j])`. Returns `None` if no such pair exists.
pub fn find_sep_violation(
    samples: &[Conversation],
    consumer: &dyn ConsumerFn,
) -> Option<Violation> {
    let nfs: Vec<Conversation> = samples.iter().map(|c| c.normalize()).collect();
    let responses: Vec<Vec<u8>> = samples.iter().map(|c| consumer.observe(c)).collect();
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            if nfs[i] == nfs[j] && responses[i] != responses[j] {
                return Some(Violation::Sep {
                    i,
                    j,
                    response_i: responses[i].clone(),
                    response_j: responses[j].clone(),
                });
            }
        }
    }
    None
}

/// Structured fidelity report: ε estimate over same-nf pairs, counts of blind/sep
/// violations, and per-violation Φ context (which coordinates differ at the violating pair).
#[derive(Debug, Clone, PartialEq)]
pub struct FidelityReport {
    pub epsilon: f64,
    pub blind_violations: usize,
    pub sep_violations: usize,
    /// For each sep violation found, the Φ coordinates of `samples[i]` minus
    /// `samples[j]` (best-effort diagnostic — which coordinate "should have" but did not
    /// distinguish them; for sep both are equal so the diff is all-zero, kept as a slot
    /// for downstream reports).
    pub sep_phi_diffs: Vec<(usize, usize, PhiDiff)>,
}

/// Pointwise Φ coordinate diff (signed). Used for reporting only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhiDiff {
    pub d_g: i64,
    pub d_sigma: i64,
    pub d_r: i64,
}

impl PhiDiff {
    pub fn between(a: PhiCoords, b: PhiCoords) -> Self {
        Self {
            d_g: a.phi_g as i64 - b.phi_g as i64,
            d_sigma: a.phi_sigma as i64 - b.phi_sigma as i64,
            d_r: a.phi_r as i64 - b.phi_r as i64,
        }
    }
}

// ----------------------------------------------------------------------------
// Structural conformance gates (需求表 v1.4 §3) — no consumer needed.
// ----------------------------------------------------------------------------

/// A structural conformance finding over the IR — a property a faithful IR/codec must
/// *surface* (for typed accounting) rather than silently repair (Reasonix-style). Unlike
/// [`Violation`], these gates need no `ConsumerFn`: they are pure structural invariants from
/// 需求表 v1.4 §3 ("绝不静默丢" red line). An empty result ⟹ the gate passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralFinding {
    /// #20 orphan-toolmsg: a `ToolResult` whose `ref` matches no `ToolCall` id anywhere in
    /// the conversation. A faithful codec surfaces the orphan into the IR (so it can be
    /// typed-accounted); a Reasonix-style codec silently drops it. This gate *reports* the
    /// orphan — distinct from [`Conversation::validate`], which *rejects* a non-closed
    /// conversation outright (strict well-formedness). Closure is order-independent here
    /// (ordering is a separate defect, #3).
    OrphanToolResult { turn_index: usize, ref_id: String },
    /// #19 interruption-recovery: a `ToolCall` that was abandoned mid-conversation — its
    /// `id` matches no `ToolResult` `ref` AND it is not in the final turn (a call in the
    /// final turn is a legitimate *pending* call awaiting execution, not an interruption).
    /// A faithful codec surfaces the abandoned call honestly; a Reasonix-style codec
    /// fabricates a synthetic `interruptedToolResult` to make the call appear answered,
    /// hiding the interruption. This gate detects the honest state so fabrication (which
    /// would empty it) is distinguishable.
    AbandonedToolCall { turn_index: usize, call_id: String },
}

/// #20 orphan-toolmsg gate. Returns every `ToolResult` whose `ref` has no matching
/// `ToolCall` id anywhere in `conv`. Empty ⟹ pass. The gate exists so a codec that *drops*
/// an orphan (Reasonix-style) is distinguishable from one that *surfaces* it: feed a native
/// carrying a known orphan through `up`, then assert this gate still finds it in the IR (a
/// dropping codec would yield an empty IR and silently "pass" the wrong way — the conformance
/// test pins the surfaced count against the native count).
pub fn find_orphan_toolresults(conv: &Conversation) -> Vec<StructuralFinding> {
    let mut call_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for turn in &conv.turns {
        for c in &turn.content {
            if let Content::ToolCall { id, .. } = c {
                call_ids.insert(id.as_str());
            }
        }
    }
    let mut out = Vec::new();
    for (turn_index, turn) in conv.turns.iter().enumerate() {
        for c in &turn.content {
            if let Content::ToolResult { ref_id, .. } = c {
                if !call_ids.contains(ref_id.as_str()) {
                    out.push(StructuralFinding::OrphanToolResult {
                        turn_index,
                        ref_id: ref_id.as_str().to_string(),
                    });
                }
            }
        }
    }
    out
}

/// #19 interruption-recovery gate. Returns every `ToolCall` that was abandoned
/// mid-conversation: no matching `ToolResult` `ref` anywhere, AND not in the final turn (a
/// trailing call is a legitimate pending call, not an interruption). Empty ⟹ pass. The gate
/// surfaces the honest interrupted state; a codec that fabricates a synthetic result to hide
/// the interruption (Reasonix-style) would empty it — the conformance test pins that.
pub fn find_abandoned_toolcalls(conv: &Conversation) -> Vec<StructuralFinding> {
    let mut result_refs: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for turn in &conv.turns {
        for c in &turn.content {
            if let Content::ToolResult { ref_id, .. } = c {
                result_refs.insert(ref_id.as_str());
            }
        }
    }
    let last = conv.turns.len().saturating_sub(1);
    let mut out = Vec::new();
    for (turn_index, turn) in conv.turns.iter().enumerate() {
        if turn_index == last {
            continue; // trailing turn → pending calls are legitimate, not interruptions
        }
        for c in &turn.content {
            if let Content::ToolCall { id, .. } = c {
                if !result_refs.contains(id.as_str()) {
                    out.push(StructuralFinding::AbandonedToolCall {
                        turn_index,
                        call_id: id.as_str().to_string(),
                    });
                }
            }
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Response-side conformance gates (SPEC v1.7 §2bis / #16 model-identity, #17 face-purity).
// magi ruling: these observe *events* (substitution / contamination), which are NOT lossful
// transformations — so they yield `Finding`s, orthogonal to `LossObligation` (which a separate
// channel records if the event later causes actual information loss). finding ⊥ LossObligation.
// ----------------------------------------------------------------------------

/// Which family-split side a [`Finding`] is accounted against (Cor5.2). A single #17 impurity
/// can raise both a `Behav` main finding and a `Bill` side-effect finding — separate accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingScope {
    Behav,
    Bill,
}

/// A response-side conformance finding (SPEC v1.7 §2bis). Distinct from [`Violation`]
/// (consumer-based) and [`StructuralFinding`] (IR-structural): these read a [`ResponseEnvelope`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finding {
    /// #16 model-identity: the provider's `echoed_model` differs from the requested model.
    /// **Sufficiency (P)**: a trigger is hard evidence of a reroute. **Necessity (H)**: a
    /// non-trigger does NOT imply no reroute (同名异权重 / 同模型异后端 evade it) — so this
    /// finding is "充分非必要", always **high** confidence when present, never a completeness
    /// claim. Implicit reroutes are caught by other channels (behav fingerprint / latency /
    /// bill rate), not here.
    Reroute { requested_model: String, echoed_model: String },
    /// #17 face-purity main finding (behav): `usage` carries fields outside the face's native
    /// set — the face identity equivalence-class boundary is broken.
    FaceImpurity { face: String, leaked_fields: Vec<String>, scope: FindingScope },
    /// #17 face-purity bill side-effect (Cor5.2): some leaked field is a usage field of *some*
    /// face, so it may pollute bill computation. Co-exists with [`Finding::FaceImpurity`] —
    /// it does not replace it (two-family separate accounting).
    ForeignUsageField { face: String, polluted_fields: Vec<String>, scope: FindingScope },
}

/// `NATIVE_USAGE_KEYS[face]` (STRICT v0, SPEC v1.7 §2bis). A (J/data) engineering allowlist —
/// the official-doc-anchored token fields native to each face. It is the engineering
/// approximation of `π_usage(≈_𝒪|face)`; maintained here, evolves with provider docs. Unknown
/// face ⟹ empty (every field then reads as foreign — conservative).
fn native_usage_keys(face: &str) -> &'static [&'static str] {
    match face {
        "anthropic" => &[
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ],
        "openai" => &["prompt_tokens", "completion_tokens", "total_tokens"],
        "gemini" => &["promptTokenCount", "candidatesTokenCount", "totalTokenCount"],
        "responses" => &["input_tokens", "output_tokens", "total_tokens"],
        _ => &[],
    }
}

/// Whether `key` is a usage field native to *some* face — used to decide the #17 bill
/// side-effect (a leaked field that is a usage field elsewhere may pollute bill computation).
fn is_known_usage_field(key: &str) -> bool {
    ["anthropic", "openai", "gemini", "responses"]
        .iter()
        .any(|f| native_usage_keys(f).contains(&key))
}

/// #16 model-identity gate (SPEC v1.7 §2bis). `echoed_model ≠ requested_model ⟹ Reroute`.
/// Sufficiency (P): a trigger is hard evidence. Necessity (H): a non-trigger is NOT proof of
/// no reroute — "充分非必要". `None` ⟹ pass (or no echo to compare).
pub fn check_model_identity(requested_model: &str, resp: &ResponseEnvelope) -> Option<Finding> {
    match &resp.echoed_model {
        Some(echoed) if echoed != requested_model => Some(Finding::Reroute {
            requested_model: requested_model.to_string(),
            echoed_model: echoed.clone(),
        }),
        _ => None,
    }
}

/// #17 face-purity gate (SPEC v1.7 §2bis). Returns the behav main finding when `usage` carries
/// any field outside the face's native set, plus a bill side-effect finding (Cor5.2) when a
/// leaked field is a usage field of some other face. Empty ⟹ pass.
pub fn check_face_purity(face: &str, resp: &ResponseEnvelope) -> Vec<Finding> {
    let native = native_usage_keys(face);
    let mut leaked: Vec<String> = resp
        .usage
        .keys()
        .filter(|k| !native.contains(&k.as_str()))
        .cloned()
        .collect();
    leaked.sort(); // BTreeMap keys already sorted, but pin determinism explicitly
    if leaked.is_empty() {
        return Vec::new();
    }
    let mut findings = vec![Finding::FaceImpurity {
        face: face.to_string(),
        leaked_fields: leaked.clone(),
        scope: FindingScope::Behav,
    }];
    let polluted: Vec<String> = leaked
        .into_iter()
        .filter(|k| is_known_usage_field(k))
        .collect();
    if !polluted.is_empty() {
        findings.push(Finding::ForeignUsageField {
            face: face.to_string(),
            polluted_fields: polluted,
            scope: FindingScope::Bill,
        });
    }
    findings
}

/// Build a fidelity report over a sample set against a consumer.
pub fn fidelity_report(samples: &[Conversation], consumer: &dyn ConsumerFn) -> FidelityReport {
    let nfs: Vec<Conversation> = samples.iter().map(|c| c.normalize()).collect();
    let responses: Vec<Vec<u8>> = samples.iter().map(|c| consumer.observe(c)).collect();

    // Build (m1, m2, ResponseVector) triples for the consumer "self".
    let mut pairs = Vec::new();
    let mut sep_phi_diffs = Vec::new();
    let mut blind_count = 0usize;
    let mut sep_count = 0usize;

    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let same_nf = nfs[i] == nfs[j];
            let same_resp = responses[i] == responses[j];
            if same_nf && !same_resp {
                sep_count += 1;
                let pi = phi_of(&samples[i], &[]);
                let pj = phi_of(&samples[j], &[]);
                sep_phi_diffs.push((i, j, PhiDiff::between(pi, pj)));
            }
            if !same_nf && same_resp {
                blind_count += 1;
            }
            if same_nf {
                let mut rv = ResponseVector::new();
                rv.insert("self", responses[i].clone(), responses[j].clone());
                pairs.push((samples[i].clone(), samples[j].clone(), rv));
            }
        }
    }

    let epsilon = epsilon_estimate(&pairs);
    FidelityReport {
        epsilon,
        blind_violations: blind_count,
        sep_violations: sep_count,
        sep_phi_diffs,
    }
}
