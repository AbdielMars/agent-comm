//! Engineering metrics over the IR.
//!
//! - `PhiCoords` — three-coordinate complexity (G/Σ/R per Mars 2026 Paper 1 primitives).
//! - `phi_of(conv, translate_chain)` — full Φ coordinates including translation overhead.
//! - `phi_msg(conv)` — per-message complexity (G + R, no translation).
//! - `epsilon_estimate(pairs)` — empirical estimate of the consumer-side faithfulness
//!   measure ε from observed response pairs over same-normal-form messages.

use crate::{codec_for, Content, Conversation, PhiKind};
use std::collections::HashMap;

/// Φ three-coordinate complexity.
/// `phi_g` — Genesis: forward generators inside one fiber.
/// `phi_sigma` — Stratification: cross-fiber translation steps (changes in provider id
///   along the translate chain).
/// `phi_r` — Reification: tool_call + tool_result generators, plus nested payload depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhiCoords {
    pub phi_g: u64,
    pub phi_sigma: u64,
    pub phi_r: u64,
}

impl PhiCoords {
    pub fn total(&self) -> u64 {
        self.phi_g + self.phi_sigma + self.phi_r
    }
}

/// Compute Φ coordinates for a conversation under an optional translation chain.
/// The chain is a sequence of provider aliases (e.g. `["claude", "openai", "gemini"]`);
/// Σ counts provider-id changes along the chain (alias normalization aware).
pub fn phi_of(conv: &Conversation, translate_chain: &[&str]) -> PhiCoords {
    let nf = conv.normalize();
    let mut g: u64 = 0;
    let mut r: u64 = 0;
    for turn in &nf.turns {
        for c in &turn.content {
            match c.phi_kind() {
                PhiKind::Genesis => g += 1,
                PhiKind::Reification => {
                    r += 1;
                    if let Content::ToolResult { payload, .. } = c {
                        r += nested_depth(payload) as u64;
                    }
                }
            }
        }
    }
    let sigma = translate_chain
        .windows(2)
        .filter(|p| codec_id_for(p[0]) != codec_id_for(p[1]))
        .count() as u64;
    PhiCoords {
        phi_g: g,
        phi_sigma: sigma,
        phi_r: r,
    }
}

/// Per-message complexity: Φ_G + Φ_R, ignoring translation overhead.
pub fn phi_msg(conv: &Conversation) -> u64 {
    let p = phi_of(conv, &[]);
    p.phi_g + p.phi_r
}

fn nested_depth(payload: &[Content]) -> usize {
    let mut max_d = 0;
    for c in payload {
        if let Content::ToolResult { payload: inner, .. } = c {
            let d = 1 + nested_depth(inner);
            if d > max_d {
                max_d = d;
            }
        }
    }
    max_d
}

/// Resolve a provider alias to its canonical id (e.g. `claude → anthropic`). Unknown
/// providers compare by raw string.
fn codec_id_for(provider: &str) -> String {
    match codec_for(provider) {
        Some(c) => c.provider_id().to_string(),
        None => provider.to_string(),
    }
}

// ============================================================================
// Empirical ε
// ============================================================================

pub type ObserverId = String;

/// Per-pair response vector: each observer records its responses to the two messages of
/// the pair. ε is estimated from the rate of disagreement across pairs whose IR normal
/// forms are equal.
#[derive(Debug, Clone, Default)]
pub struct ResponseVector {
    pub responses: HashMap<ObserverId, (Vec<u8>, Vec<u8>)>,
}

impl ResponseVector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, observer: impl Into<ObserverId>, r1: Vec<u8>, r2: Vec<u8>) {
        self.responses.insert(observer.into(), (r1, r2));
    }
}

/// Empirical ε estimate: `sup_o Pr_{same nf}[response_o(m1) != response_o(m2)]`.
///
/// Inputs: triples `(m1, m2, response_vector)` of paired conversations and the recorded
/// downstream response for each observer. Pairs whose normal forms are not equal are
/// skipped (ε is only defined over same-nf samples). Returns 0.0 if there are no
/// qualifying samples; returns the supremum over observers otherwise.
pub fn epsilon_estimate(pairs: &[(Conversation, Conversation, ResponseVector)]) -> f64 {
    let mut by_observer: HashMap<ObserverId, (u64, u64)> = HashMap::new();
    for (m1, m2, resp) in pairs {
        if m1.normalize() != m2.normalize() {
            continue;
        }
        for (o_id, (r1, r2)) in &resp.responses {
            let entry = by_observer.entry(o_id.clone()).or_insert((0, 0));
            entry.0 += 1; // total
            if r1 != r2 {
                entry.1 += 1; // different
            }
        }
    }
    by_observer
        .values()
        .map(|(t, d)| if *t == 0 { 0.0 } else { *d as f64 / *t as f64 })
        .fold(0.0f64, f64::max)
}
