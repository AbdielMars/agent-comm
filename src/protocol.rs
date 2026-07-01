//! Protocol contract (inlined from agent-protocol).
//!
//! Minimal contract: every guarded operation carries a [`Principal`] and consults an
//! [`IdentityGate`]. Only the **contract** lives here — no gate implementation is provided.
//! The gate is consulted **fail-closed**: a guarded operation proceeds only if the gate
//! authorizes the principal; otherwise it must return an error (never a silent default).

/// Opaque actor identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Principal(pub String);

impl Principal {
    pub fn new(id: impl Into<String>) -> Self {
        Principal(id.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Identity gate consulted by every guarded operation.
///
/// Implementations decide what "authorized" and "stable" mean for their deployment.
pub trait IdentityGate {
    /// Is this principal authorized to act?
    fn verify(&self, who: &Principal) -> bool;

    /// Does this principal's identity remain stable under transformation?
    fn preserved(&self, who: &Principal) -> bool;
}
