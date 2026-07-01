//! # agent-comm — a neutral conversation IR with a normal form and provider codecs
//!
//! A single neutral conversation type [`Conversation`] is a **colimit over a small kernel K**
//! of content generators ([`Content::Text`] / [`Content::ToolCall`] / [`Content::ToolResult`]).
//! Every provider format converts to/from this IR via a [`ProviderCodec`] — never pairwise
//! between providers.
//!
//! [`Conversation::normalize`] puts a conversation into a **normal form** (idempotent). The
//! full pipeline (R5 → R1 → R6 → R3 → R2) lands incrementally; v0 currently runs the R5/R1/R3
//! prefix. Provider conversion that cannot carry a feature records a [`LossObligation`]
//! (never silently dropped) — two-sided: both `up` and `down` may report loss (R-3 invariant).
//!
//! [`encode`] / [`decode`] are **fail-closed**: each requires an authorized [`Principal`]
//! via an [`IdentityGate`]; unauthorized → `Err(CommError::Unauthorized)`, no bypass.
//!
//! Extension generators ([`Content::Thinking`] / [`Content::Media`] / [`Content::Video`])
//! are **wire-frozen** (SPEC v1.7 §2bis). All six kernels (Text/ToolCall/ToolResult +
//! Thinking/Media/Video) share a stable serialization; provider codecs that cannot represent
//! an extension record a typed [`LossObligation`] (R-3 never silent) rather than failing.

use crate::protocol::{IdentityGate, Principal};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub mod check;
pub mod codecs;
pub mod conformance;
pub mod metrics;
pub mod protocol;

// ============================================================================
// Role
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// Canonicalize a vendor role alias to the kernel role. v0 covers the most common
    /// aliases; the full alias table is an extension point.
    pub fn canon(raw: &str) -> Option<Role> {
        match raw.to_ascii_lowercase().as_str() {
            "system" | "developer" => Some(Role::System),
            "user" | "human" => Some(Role::User),
            "assistant" | "ai" | "model" => Some(Role::Assistant),
            "tool" | "function" => Some(Role::Tool),
            _ => None,
        }
    }
}

// ============================================================================
// Content generators (kernel K + extension generators)
// ============================================================================

pub type CallId = String;
pub type ToolName = String;

/// Content generators. Kernel K = `Text` / `ToolCall` / `ToolResult` (wire frozen).
/// Extending = add a variant = add a leg to the colimit; old legs unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
        /// Optional prompt-cache breakpoint on this block (SPEC v1.7 / G3). Carried in the
        /// IR (∈ ≈_bill) so a same-vendor round-trip is lossless; a vendor that cannot
        /// express it drops it → typed `bill.cache_directive_lost` (Cor5.2: billed against
        /// ≈_bill only). Omitted from the wire when `None` (back-compat).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolCall {
        id: CallId,
        name: ToolName,
        /// Tool arguments. Object key order is vendor-accidental; the normal form sorts
        /// keys via R2 (`r2_args_meta_sort`, extension point).
        args: Value,
    },
    ToolResult {
        #[serde(rename = "ref")]
        ref_id: CallId,
        /// Restricted nesting (N1): payload must not contain `ToolCall` / `Thinking` /
        /// `Video`; nesting depth ≤ [`NESTING_LIMIT_D0`].
        payload: Vec<Content>,
        /// Optional prompt-cache breakpoint on this block (SPEC v1.7 / G3). See `Text`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Reasoning / chain-of-thought (extension generator).
    /// `sig` carries a vendor-accidental signature (e.g. Anthropic `thinking.signature`,
    /// Gemini `thoughtSignature`) — emitted only by `down` for the originating vendor;
    /// not part of cross-vendor semantics. Loss to a vendor without a signature channel
    /// is reported as a [`LossObligation`] (R-3).
    ///
    /// `placement` (SPEC v1.7 / G1) records WHERE the reasoning sits in the native wire —
    /// a dedicated block, embedded in `tool_call` kwargs (e.g. Letta
    /// `put_inner_thoughts_in_kwargs`), or an inline string (e.g. DeepSeek
    /// `reasoning_content`). placement ∈ ≈_𝒪 (magi G1 ruling: Letta runtime tool dispatch
    /// depends on it → representation-NOT-blind), so it is carried in the IR; a codec that
    /// collapses placement must record a typed `behav.placement_collapsed` loss.
    ///
    /// **Wire-frozen** (SPEC v1.7 §2bis). A provider codec that cannot carry reasoning
    /// records a `behav.placement_collapsed` / `thinking` / `thinking+signature` loss.
    Thinking {
        text: String,
        #[serde(rename = "sig", skip_serializing_if = "Option::is_none", default)]
        sig: Option<String>,
        /// Where the reasoning sits in the native wire (G1). Defaults to `Block` for
        /// back-compat with pre-v1.7 serialized data.
        #[serde(default, skip_serializing_if = "Placement::is_default")]
        placement: Placement,
    },
    /// Multimodal media (extension generator): inline base64 data + MIME (image/* or
    /// application/pdf). N1: permitted inside `tool_result.payload`.
    ///
    /// **Wire-frozen** (SPEC v1.7 §2bis).
    Media {
        mime: String,
        data: String,
    },
    /// Video media (extension generator). `source` supports URL (videos are often too large
    /// to inline) or base64. N1: not permitted inside `tool_result.payload`.
    ///
    /// **Wire-frozen** (SPEC v1.7 §2bis). N1 restricts this variant: not permitted inside
    /// `tool_result.payload`.
    Video {
        source: VideoSource,
        mime: String,
        #[serde(
            rename = "duration_seconds",
            skip_serializing_if = "Option::is_none",
            default
        )]
        duration_seconds: Option<u64>,
    },
}

/// Video payload source. Internal tag `src` — wire form:
/// `{"src":"url","url":"..."}` / `{"src":"base64","data":"..."}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "src", rename_all = "snake_case")]
pub enum VideoSource {
    Url { url: String },
    Base64 { data: String },
}

/// Reasoning placement (SPEC v1.7 / G1): where chain-of-thought sits in the native wire.
/// `placement ∈ ≈_𝒪` (magi G1 ruling) — carried in the IR, never silently normalized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// A dedicated reasoning block (e.g. Anthropic `thinking`). Default for back-compat.
    #[default]
    Block,
    /// Reasoning embedded in tool-call arguments (e.g. Letta `put_inner_thoughts_in_kwargs`).
    ToolCallKwargs,
    /// Reasoning as an inline string field (e.g. DeepSeek `reasoning_content`).
    InlineString,
}

impl Placement {
    /// `skip_serializing_if` helper: omit `placement` from the wire when it is the default
    /// (`Block`), keeping pre-v1.7 round-trips byte-stable.
    pub fn is_default(&self) -> bool {
        matches!(self, Placement::Block)
    }

    /// Stable lowercase wire token, used in `behav.placement_collapsed` loss details.
    pub fn as_str(&self) -> &'static str {
        match self {
            Placement::Block => "block",
            Placement::ToolCallKwargs => "tool_call_kwargs",
            Placement::InlineString => "inline_string",
        }
    }
}

/// Cache directive (SPEC v1.7 / G3): a client-controlled prompt-cache breakpoint on a
/// content block. `cache_control ∈ ≈_bill` (magi G3 ruling: controls cache-hit pricing) and
/// `key` depends on the kernel generators, so it is carried in the IR (not the envelope); a
/// codec that drops it must record a typed `bill.cache_directive_lost` loss.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    /// Client-controlled cache scope id (optional; vendors that auto-scope omit it).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub key: Option<String>,
    /// Whether caching is requested at this breakpoint.
    pub enabled: bool,
    /// Time-to-live: a vendor token (`"ephemeral"` / `"1h"`) or a second count as a string.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ttl: Option<String>,
}

/// Cache freshness signal (SPEC v1.7 / G9): the consistency guarantee of a replayed
/// idempotent response. `unknown` is the conservative default (no provider freshness signal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheFreshness {
    /// Provider confirmed the cached response still matches current ground truth.
    Fresh,
    /// Provider flagged the cache expired but served it anyway (weak guarantee).
    Stale,
    /// No freshness signal — conservative default, weakest guarantee.
    #[default]
    Unknown,
}

/// Maximum nesting depth for `tool_result.payload` (N1 invariant). v0 = 2.
pub const NESTING_LIMIT_D0: usize = 2;

/// Phi-coordinate kind (G / R per Mars 2026 Paper 1 primitives).
/// `Genesis` = forward-direction generator inside one fiber.
/// `Reification` = tool-call / tool-result pair (ρ-lift).
/// Stratification (Σ) — cross-fiber translation step count — is not in this enum;
/// it is computed by `metrics::phi_of` from the translation chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiKind {
    Genesis,
    Reification,
}

impl Content {
    /// N1 validation: `tool_result.payload` must not contain `tool_call` / `thinking` /
    /// `video`; nesting depth ≤ [`NESTING_LIMIT_D0`].
    pub fn validate(&self, depth: usize) -> Result<(), CodecError> {
        if let Content::ToolResult { payload, .. } = self {
            if depth + 1 > NESTING_LIMIT_D0 {
                return Err(CodecError::Malformed(format!(
                    "tool_result nesting exceeds D0={NESTING_LIMIT_D0}"
                )));
            }
            for p in payload {
                if matches!(
                    p,
                    Content::ToolCall { .. } | Content::Thinking { .. } | Content::Video { .. }
                ) {
                    return Err(CodecError::Malformed(
                        "tool_call/thinking/video nested inside tool_result (forbidden by N1)"
                            .into(),
                    ));
                }
                p.validate(depth + 1)?;
            }
        }
        Ok(())
    }

    /// Phi-coordinate classification (Genesis / Reification). Consumed by the `metrics`
    /// module.
    pub(crate) fn phi_kind(&self) -> PhiKind {
        match self {
            Content::Text { .. }
            | Content::Thinking { .. }
            | Content::Media { .. }
            | Content::Video { .. } => PhiKind::Genesis,
            Content::ToolCall { .. } | Content::ToolResult { .. } => PhiKind::Reification,
        }
    }
}

// ============================================================================
// Turn / Conversation
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub role: Role,
    pub content: Vec<Content>,
}

/// A well-formed conversation. `Default::default()` = empty conversation (initial object).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Conversation {
    pub turns: Vec<Turn>,
}

impl Conversation {
    /// Structural well-formedness:
    /// - N1 (per-content validation, see [`Content::validate`]);
    /// - tool-link closure: every `tool_result.ref` points to a preceding `tool_call.id`;
    /// - `ToolCall` id uniqueness (R3 and ref resolution both rely on uniqueness —
    ///   duplicates would fuse distinct calls).
    pub fn validate(&self) -> Result<(), CodecError> {
        let mut seen_calls: Vec<&str> = Vec::new();
        for turn in &self.turns {
            for c in &turn.content {
                c.validate(0)?;
                match c {
                    Content::ToolCall { id, .. } => {
                        if seen_calls.contains(&id.as_str()) {
                            return Err(CodecError::Malformed(format!(
                                "duplicate tool_call id '{id}' (ids must be unique)"
                            )));
                        }
                        seen_calls.push(id);
                    }
                    Content::ToolResult { ref_id, .. } => {
                        if !seen_calls.contains(&ref_id.as_str()) {
                            return Err(CodecError::Malformed(format!(
                                "tool_result ref '{ref_id}' has no preceding tool_call"
                            )));
                        }
                    }
                    Content::Text { .. }
                    | Content::Thinking { .. }
                    | Content::Media { .. }
                    | Content::Video { .. } => {}
                }
            }
        }
        Ok(())
    }

    /// The unique normal form nf(·). Two conversations with the same semantics share the
    /// same normal form (strict equality). Idempotence — necessary for confluence (T1) —
    /// is guarded by the `normalize_is_idempotent` test.
    ///
    /// Pipeline order is fixed for determinism:
    ///   R5 drop-empty → R1 text-merge → R6 fold tool_result into Tool turns
    ///   → R3 global id-canon (role-independent, placed late) → R2 args key sort.
    pub fn normalize(&self) -> Conversation {
        let mut c = self.clone();
        for turn in &mut c.turns {
            let items = std::mem::take(&mut turn.content);
            let items = r5_empty_elim(items); // R5
            turn.content = r1_text_merge(items); // R1
        }
        r6_tool_result_role_canon(&mut c); // R6
        r3_id_canon(&mut c); // R3
        r2_args_meta_sort(&mut c); // R2
        c
    }

    /// Two conversations are semantically equal in the IR iff their normal forms are equal.
    pub fn semantic_eq(&self, other: &Conversation) -> bool {
        self.normalize() == other.normalize()
    }
}

// ============================================================================
// Normal form pipeline (R5 → R1 → R6 → R3 → R2)
// ============================================================================

/// R5: drop empty text blocks (recurses into `tool_result.payload`).
fn r5_empty_elim(items: Vec<Content>) -> Vec<Content> {
    items
        .into_iter()
        .filter_map(|c| match c {
            Content::Text { text, .. } if text.is_empty() => None,
            Content::ToolResult { ref_id, payload, cache_control } => Some(Content::ToolResult {
                ref_id,
                payload: r5_empty_elim(payload),
                cache_control,
            }),
            other => Some(other),
        })
        .collect()
}

/// R1: merge adjacent text blocks (recurses into `tool_result.payload`).
fn r1_text_merge(items: Vec<Content>) -> Vec<Content> {
    let mut out: Vec<Content> = Vec::with_capacity(items.len());
    for item in items {
        let item = match item {
            Content::ToolResult { ref_id, payload, cache_control } => Content::ToolResult {
                ref_id,
                payload: r1_text_merge(payload),
                cache_control,
            },
            other => other,
        };
        match (out.last_mut(), &item) {
            // Merge adjacent text only when the cache breakpoint matches — a different
            // `cache_control` is a cache boundary (G3) that must not be fused away.
            (
                Some(Content::Text { text: prev, cache_control: pcc }),
                Content::Text { text: cur, cache_control: ccc },
            ) if pcc == ccc => {
                prev.push_str(cur);
            }
            _ => out.push(item),
        }
    }
    out
}

/// R3: rename `CallId`s in first-appearance order to `call_0, call_1, ...`.
fn r3_id_canon(c: &mut Conversation) {
    let mut map: HashMap<CallId, CallId> = HashMap::new();
    let mut counter: u64 = 0;
    for turn in &c.turns {
        for content in &turn.content {
            collect_call_ids(content, &mut map, &mut counter);
        }
    }
    for turn in &mut c.turns {
        for content in &mut turn.content {
            rewrite_ids(content, &map);
        }
    }
}

fn collect_call_ids(c: &Content, map: &mut HashMap<CallId, CallId>, counter: &mut u64) {
    match c {
        Content::ToolCall { id, .. } => {
            map.entry(id.clone()).or_insert_with(|| {
                let n = format!("call_{counter}");
                *counter += 1;
                n
            });
        }
        // N1 forbids `ToolCall` inside `tool_result.payload` at any depth, so there is
        // nothing to collect here. We do NOT recurse into payload: for valid input it would
        // be dead code; for invalid input it would pollute the R3 counter.
        Content::ToolResult { .. } => {}
        Content::Text { .. }
        | Content::Thinking { .. }
        | Content::Media { .. }
        | Content::Video { .. } => {}
    }
}

fn rewrite_ids(c: &mut Content, map: &HashMap<CallId, CallId>) {
    match c {
        Content::ToolCall { id, .. } => {
            if let Some(n) = map.get(id) {
                *id = n.clone();
            }
        }
        Content::ToolResult { ref_id, payload, .. } => {
            if let Some(n) = map.get(ref_id) {
                *ref_id = n.clone();
            }
            for p in payload {
                rewrite_ids(p, map);
            }
        }
        Content::Text { .. }
        | Content::Thinking { .. }
        | Content::Media { .. }
        | Content::Video { .. } => {}
    }
}

/// R6: tool_result-role canonicalization. Once a `tool_result` is anchored to its
/// `tool_call` via `ref_id`, the semantics is independent of the role of the carrier turn
/// (some providers emit `user`, some `tool`, some other — vendor-accidental). The normal
/// form eliminates this: each `ToolResult` is hoisted into its own `Role::Tool` turn
/// (preserving order); non-tool_result content stays in the original-role turn.
///
/// Post-condition: after R6 every `ToolResult` lives in a `Tool` turn that contains
/// exactly one item. Cross-vendor consistency: `user[tr1, tr2]` (one provider) and
/// `tool[tr1], tool[tr2]` (another) both normalize to `Tool[tr1], Tool[tr2]`.
///
/// Design rationale: turns whose content is empty after R5/R1 (and which carry no
/// tool_result) are dropped on reassembly — this is the turn-level promotion of R5's
/// "drop empty" spirit, an intentional step (locked in by the `r6_drops_empty_turns` test),
/// not a side effect.
fn r6_tool_result_role_canon(c: &mut Conversation) {
    let mut new_turns: Vec<Turn> = Vec::new();
    for turn in std::mem::take(&mut c.turns) {
        let role = turn.role;
        let mut acc: Vec<Content> = Vec::new();
        for item in turn.content {
            if matches!(item, Content::ToolResult { .. }) {
                // Flush accumulated non-tool_result content (keeps original role), then
                // emit a dedicated Tool turn for this single tool_result.
                if !acc.is_empty() {
                    new_turns.push(Turn {
                        role,
                        content: std::mem::take(&mut acc),
                    });
                }
                new_turns.push(Turn {
                    role: Role::Tool,
                    content: vec![item],
                });
            } else {
                acc.push(item);
            }
        }
        if !acc.is_empty() {
            new_turns.push(Turn { role, content: acc });
        }
    }
    c.turns = new_turns;
}

/// R2: args meta-sort. `ToolCall.args` object key order is vendor-accidental (preserved
/// by serde_json's `preserve_order`). The normal form recursively sorts object keys; array
/// order is preserved (array order is semantic).
fn r2_args_meta_sort(c: &mut Conversation) {
    for turn in &mut c.turns {
        for content in &mut turn.content {
            canon_args(content);
        }
    }
}

fn canon_args(c: &mut Content) {
    match c {
        Content::ToolCall { args, .. } => *args = canon_value(args),
        Content::ToolResult { payload, .. } => {
            for p in payload {
                canon_args(p);
            }
        }
        Content::Text { .. }
        | Content::Thinking { .. }
        | Content::Media { .. }
        | Content::Video { .. } => {}
    }
}

/// Recursively canonicalize JSON object key order (via `BTreeMap` rebuild, independent of
/// the underlying Map implementation).
fn canon_value(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let sorted: std::collections::BTreeMap<String, Value> = map
                .iter()
                .map(|(k, val)| (k.clone(), canon_value(val)))
                .collect();
            serde_json::to_value(sorted).unwrap_or_else(|_| v.clone())
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canon_value).collect()),
        other => other.clone(),
    }
}

// ============================================================================
// Loss accounting (R-3 invariant: both `up` and `down` may report loss; never silent)
// ============================================================================

/// A feature a provider conversion could not carry, recorded as a typed obligation.
/// Two-sided: both `up` (lifting native → IR) and `down` (lowering IR → native) may emit loss.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LossObligation {
    pub provider: String,
    /// Dropped generator kind — the concrete typed subtype. Existing values:
    /// `"thinking"` / `"media"` / `"video"` / `"system_position"` / `"intra_turn_order"` /
    /// `"thinking.segment_boundaries"` / `"tool_result.dangling_ref"` / `"unknown_block:…"`.
    ///
    /// SPEC v1.7 / magi v0.3 taxonomy (9 LossObligation families): truncation / translation /
    /// role_collapse / signature / tool_id / media / retry_side_effect / canonicalization /
    /// **stale_cache_response** (9th, G9). Two new typed payloads ride existing families:
    /// `"behav.placement_collapsed"` (G1, translation) and `"bill.cache_directive_lost"`
    /// (G3, translation). A `bill.`-prefixed kind is billed against ≈_bill only (Cor5.2
    /// family split), a `behav.`-prefixed kind against ≈_behav only.
    pub dropped_kind: String,
    /// Turn index where the loss occurred (best-effort).
    pub turn_index: usize,
    /// Whether the dropped feature is recoverable on the return trip.
    pub recoverable: bool,
    pub note: String,
    /// Structured typed payload for the loss (SPEC v1.7 / magi v0.3). Rides the existing
    /// loss families (path A — no new subtypes): e.g. `behav.placement_collapsed` carries
    /// `{from_placement, to_placement}`, `bill.cache_directive_lost` carries `{lost_field}`.
    /// `None` for legacy losses. A `BTreeMap<String,String>` keeps it `Eq` + deterministic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<std::collections::BTreeMap<String, String>>,
}

impl LossObligation {
    /// Constructor used by codecs to emit a typed loss record (R-3 accounting).
    pub fn new(
        provider: &str,
        dropped_kind: &str,
        turn_index: usize,
        recoverable: bool,
        note: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            dropped_kind: dropped_kind.into(),
            turn_index,
            recoverable,
            note: note.into(),
            detail: None,
        }
    }

    /// Constructor that attaches a structured typed payload (SPEC v1.7 / magi v0.3).
    /// `detail` carries the subtype-specific fields (e.g. `from_placement`/`to_placement`
    /// for `behav.placement_collapsed`, `lost_field` for `bill.cache_directive_lost`).
    pub fn with_detail(
        provider: &str,
        dropped_kind: &str,
        turn_index: usize,
        recoverable: bool,
        note: impl Into<String>,
        detail: std::collections::BTreeMap<String, String>,
    ) -> Self {
        Self {
            detail: Some(detail),
            ..Self::new(provider, dropped_kind, turn_index, recoverable, note)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// Input does not match the provider format or violates an IR well-formedness rule.
    Malformed(String),
    /// A kernel generator that this provider cannot express (should be paired with a
    /// [`LossObligation`], not silently dropped).
    Unsupported(String),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::Malformed(s) => write!(f, "malformed: {s}"),
            CodecError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}
impl std::error::Error for CodecError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommError {
    /// The identity gate did not authorize the principal (fail-closed; no bypass).
    Unauthorized,
    Codec(CodecError),
}

// ============================================================================
// Provider codec trait (a pair of functors `up_p` / `down_p` over the kernel)
// ============================================================================

/// One pair `(up_p, down_p)` per provider; the IR is their common codomain.
///
/// Contract:
/// - `up` is full+faithful-targeted: when a native structure cannot be expressed in the
///   IR, it must emit a [`LossObligation`] (never silently dropped).
/// - Round-trip: `down ∘ up` is semantically identity on the kernel K, verified at the
///   normal-form level (`up → normalize → down → up → normalize` — same nf).
/// - Both `up` and `down` may report loss (R-3 two-sided accounting).
pub trait ProviderCodec {
    fn provider_id(&self) -> &'static str;
    fn up(&self, native: &Value) -> Result<(Conversation, Vec<LossObligation>), CodecError>;
    fn down(&self, conv: &Conversation) -> Result<(Value, Vec<LossObligation>), CodecError>;
}

/// Round-trip conformance check (engineering dual of the round-trip retraction property):
/// `up → normalize → down → up → normalize`; the two normal forms must be equal. Strict
/// `native == native2` would be too strong (JSON ordering / whitespace noise), so equality
/// is verified at the normal-form level.
pub fn check_round_trip<C: ProviderCodec + ?Sized>(
    codec: &C,
    native: &Value,
) -> Result<bool, CodecError> {
    let (c1, _up_loss) = codec.up(native)?;
    let c1 = c1.normalize();
    c1.validate()?;
    let (native2, _down_loss) = codec.down(&c1)?;
    let (c2, _up_loss2) = codec.up(&native2)?;
    let c2 = c2.normalize();
    Ok(c1 == c2)
}

/// Lower the IR to a provider's native format. Fail-closed on identity; normalizes first.
pub fn encode(
    conv: &Conversation,
    codec: &dyn ProviderCodec,
    who: &Principal,
    gate: &dyn IdentityGate,
) -> Result<(Value, Vec<LossObligation>), CommError> {
    if !gate.verify(who) {
        return Err(CommError::Unauthorized);
    }
    codec.down(&conv.normalize()).map_err(CommError::Codec)
}

/// Lift a provider's native format into the IR. Fail-closed on identity.
pub fn decode(
    native: &Value,
    codec: &dyn ProviderCodec,
    who: &Principal,
    gate: &dyn IdentityGate,
) -> Result<(Conversation, Vec<LossObligation>), CommError> {
    if !gate.verify(who) {
        return Err(CommError::Unauthorized);
    }
    let (conv, loss) = codec.up(native).map_err(CommError::Codec)?;
    Ok((conv.normalize(), loss))
}

/// Build a kernel text turn (convenience).
pub fn text_turn(role: Role, text: &str) -> Turn {
    Turn {
        role,
        content: vec![Content::Text { text: text.into(), cache_control: None }],
    }
}

// ============================================================================
// Codec registry + request envelope + cross-vendor translate
// ============================================================================

use codecs::{AnthropicCodec, BottomCodec, GeminiCodec, OpenAiCodec, ResponsesCodec};

/// Routing layer entry point. Alias normalization (claude→anthropic, chat→openai,
/// codex→responses). Returns a trait object — removing any one codec leaves the others
/// unaffected (providers are peers in the colimit; no privileged anchor).
pub fn codec_for(provider: &str) -> Option<Box<dyn ProviderCodec>> {
    match provider {
        "anthropic" | "claude" => Some(Box::new(AnthropicCodec)),
        // The openai chat-completions wire shape is shared by several providers
        // (deepseek, kimi, mistral via openai-compat, ...). Listed by provider name
        // for routing-layer convenience; the codec is the same.
        "openai" | "chat" | "openai_chat" | "deepseek" | "kimi" | "mistral" => {
            Some(Box::new(OpenAiCodec))
        }
        "gemini" => Some(Box::new(GeminiCodec)),
        "responses" | "codex" | "codex_chat" => Some(Box::new(ResponsesCodec)),
        // ⊥ 0-anchor (text-only floor); maximal-loss baseline, not a faithful codec.
        "bottom" => Some(Box::new(BottomCodec)),
        _ => None,
    }
}

/// Per-provider "conversation-carrying keys" — `split_envelope` removes these from the
/// native request; whatever remains is the request envelope.
fn conversation_keys(provider: &str) -> &'static [&'static str] {
    match provider {
        "anthropic" | "claude" => &["system", "messages"],
        "openai" | "chat" | "openai_chat" => &["messages"],
        "gemini" => &["systemInstruction", "system_instruction", "contents"],
        "responses" | "codex" | "codex_chat" => &["instructions", "input"],
        _ => &[],
    }
}

/// Request envelope: request-level fields outside the conversation IR (`model` /
/// `max_tokens` / `temperature` / `tools` / `tool_choice` / ...). Invariant R-2: these
/// fields never enter the IR kernel; the envelope layer reattaches them after `down`.
/// Cross-vendor parameter-name mapping (`max_tokens` ↔ `max_output_tokens`, etc.) belongs
/// to the routing layer, not to the IR.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestEnvelope {
    pub fields: serde_json::Map<String, Value>,
    /// Canonical hash of the native request body (SPEC v1.7 / G8). Pure-envelope mechanism
    /// (not in ≈_𝒪 / ≈_bill, magi G8 ruling): the precise `idempotency_violation` predicate
    /// is "same `idempotency_key` ∧ different `request_fingerprint`". Computed by
    /// [`split_envelope`] via [`request_fingerprint`].
    pub request_fingerprint: Option<String>,
    /// Cache freshness of a replayed idempotent response (SPEC v1.7 / G9). `Unknown` is the
    /// conservative default; set by response-side logic when a provider supplies a signal.
    pub cache_freshness: CacheFreshness,
}

/// Canonical request-body fingerprint (SPEC v1.7 / G8 + §4 spec canonical): canonicalize
/// object key order ([`canon_value`]), serialize deterministically, then blake3. Same body
/// under different serializations → same fingerprint.
pub fn request_fingerprint(native: &Value) -> String {
    let canon = canon_value(native);
    let bytes = serde_json::to_vec(&canon).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

/// The response-side envelope (SPEC v1.7 §2bis, magi charter-extension ruling). The dual end
/// of [`RequestEnvelope`] in a **dual envelope wrapping** (a symmetric engineering pattern —
/// NOT a categorical dual; it need not satisfy any `envelope^op` axiom, only the same envelope
/// neutrality as the request side). Carries response **metadata only** — never the response
/// content, which is already expressible as an assistant turn in the kernel-K
/// [`Conversation`]. Every field is neutral: it does not depend on K content (`echoed_model` /
/// `usage` / `stop_reason` are metadata) or depends only via a hash (`response_fingerprint`).
///
/// Inputs for the response-side conformance gates: `echoed_model` → [`check::check_model_identity`]
/// (#16), `usage` → [`check::check_face_purity`] (#17).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResponseEnvelope {
    /// Provider's self-reported model id. `None` when a provider does not echo it.
    pub echoed_model: Option<String>,
    /// Token-accounting field set. `BTreeMap` so keys are enumerable + deterministic
    /// (the #17 gate computes `keys() − NATIVE_USAGE_KEYS[face]`).
    pub usage: std::collections::BTreeMap<String, u64>,
    /// Termination reason (defect #7 canonicalization deferred — string pass-through for now).
    pub stop_reason: Option<String>,
    /// `canonical_hash(response_body)`, same normalization as [`request_fingerprint`]. `None`
    /// when the raw body is unavailable (e.g. a streaming response). Hash-only ⟹ neutral; the
    /// raw response is never stored (only hash + tokens).
    pub response_fingerprint: Option<String>,
}

/// Canonical response-body fingerprint (SPEC v1.7 §2bis): same normalization as
/// [`request_fingerprint`] (canon key order → deterministic serialize → blake3).
pub fn response_fingerprint(native: &Value) -> String {
    request_fingerprint(native)
}

/// Split a provider's native request into (conversation IR, envelope, up-side loss).
/// The conversation enters the IR; non-conversation fields stay in the envelope; any
/// dropped structure is recorded in `up_loss`.
pub fn split_envelope(
    provider: &str,
    native: &Value,
) -> Result<(Conversation, RequestEnvelope, Vec<LossObligation>), CodecError> {
    let codec = codec_for(provider)
        .ok_or_else(|| CodecError::Unsupported(format!("no codec for '{provider}'")))?;
    let (conv, up_loss) = codec.up(native)?;
    let mut fields = native.as_object().cloned().unwrap_or_default();
    for k in conversation_keys(provider) {
        fields.remove(*k);
    }
    let env = RequestEnvelope {
        fields,
        request_fingerprint: Some(request_fingerprint(native)),
        cache_freshness: CacheFreshness::Unknown,
    };
    Ok((conv, env, up_loss))
}

/// Reassemble a provider's native request from (conversation IR, envelope). `down`
/// produces the conversation structure; envelope parameters are reattached via
/// `or_insert` (conversation keys already written by `down` win; envelope only fills
/// keys it did not cover).
pub fn apply_envelope(
    provider: &str,
    conv: &Conversation,
    env: &RequestEnvelope,
) -> Result<(Value, Vec<LossObligation>), CodecError> {
    let codec = codec_for(provider)
        .ok_or_else(|| CodecError::Unsupported(format!("no codec for '{provider}'")))?;
    let (mut native, loss) = codec.down(conv)?;
    if let Some(obj) = native.as_object_mut() {
        for (k, v) in &env.fields {
            obj.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    Ok((native, loss))
}

/// Cross-vendor translation entry point: `from`-native → IR (normalized) → `to`-native +
/// merged loss list. Engineering dual of the decomposition theorem:
/// `anthropic_to_openai = translate("anthropic", "openai") = down_openai ∘ up_anthropic`.
/// Replaces the entire family of legacy pairwise `X_to_Y` converters.
///
/// Note: this translates the **conversation** only. Cross-vendor mapping of envelope
/// parameters is the routing layer's job (see `split_envelope` / `apply_envelope`).
pub fn translate(
    from: &str,
    to: &str,
    native: &Value,
) -> Result<(Value, Vec<LossObligation>), CodecError> {
    let up = codec_for(from)
        .ok_or_else(|| CodecError::Unsupported(format!("no codec for source '{from}'")))?;
    let down = codec_for(to)
        .ok_or_else(|| CodecError::Unsupported(format!("no codec for target '{to}'")))?;
    let (c, up_loss) = up.up(native)?;
    let c = c.normalize();
    c.validate()?;
    let (out, mut loss) = down.down(&c)?;
    // R-3 two-sided accounting: report both up-side and down-side loss; merge before return.
    loss.extend(up_loss);
    Ok((out, loss))
}
