# Codec Builder Guide — building an `agent-comm` compatible 插头 (adapter)

> Step2 / S2-2. How anyone builds a new codec (a 插头 / adapter between a wire format and the
> neutral IR) and earns the **"agent-comm compatible"** stamp by passing the conformance suite.
> You do not need us — the standard is the suite. Build, run `run_conformance`, ship.

---

## 0. What a codec is

A codec is one leg of a colimit: it converts a provider's native wire ⇄ the neutral
conversation IR. It never converts pairwise between providers — only to/from the IR.

```rust
pub trait ProviderCodec {
    fn provider_id(&self) -> &'static str;
    fn up(&self, native: &Value)   -> Result<(Conversation, Vec<LossObligation>), CodecError>; // wire → IR
    fn down(&self, conv: &Conversation) -> Result<(Value, Vec<LossObligation>), CodecError>;   // IR → wire
}
```

The IR kernel **K is frozen**: `Text` / `ToolCall` / `ToolResult`, plus extension generators
`Thinking` / `Media` / `Video` and the `cache_control` / `placement` fields. You map your wire
onto these — you do not invent new kernel generators.

---

## 1. The two iron rules

1. **Never drop silently (R-3).** Anything your wire cannot carry — on either `up` OR `down` —
   must be recorded as a typed [`LossObligation`], never swallowed. A silent `{}` / dropped
   block / fabricated placeholder is the single thing the standard exists to forbid.
2. **Fail closed.** Malformed input → `Err(CodecError::Malformed(..))`. Unrepresentable feature
   → a typed loss. Never guess, never paper over.

---

## 2. The 5-step build process

### Step 1 — wire 摸排 (measure, don't assume)
Capture real request/response payloads from the provider. Record the actual shapes: where do
tool calls live? tool results? reasoning? usage? Do NOT build from the API docs alone — measure.
Store schema + numbers, never raw user content.

### Step 2 — map onto kernel K + canonical 化
For each wire element, decide its kernel generator. Object key order is vendor-accidental — the
normal form sorts it; do not depend on order. Match the existing reference codecs
(`src/codecs/anthropic.rs` etc.) for idiom.

### Step 3 — LossObligation typed 记账
For every feature your wire cannot express, push a typed loss with a clear `dropped_kind` and a
`detail` map (schema + numbers only). Use the family prefix convention:
`behav.*` (≈_𝒪 / behaviour) vs `bill.*` (≈_bill / billing) — Cor5.2 keeps the two families on
separate accounts. Examples already in the tree: `behav.truncated_args`,
`behav.placement_collapsed`, `bill.cache_directive_lost`.

### Step 4 — fingerprint
The envelope layer hashes the request body with `request_fingerprint` (canonical → blake3); the
response side mirrors it with `response_fingerprint`. You normally get this for free via
`split_envelope`; do not roll your own hash.

### Step 5 — conformance 验 (the stamp)
Write vectors in YOUR wire shape and run the suite:

```rust
let report = run_conformance(&MyCodec, &my_vectors);
assert!(report.passed());        // every gate PASS ⟹ "agent-comm compatible"
```

Or `cargo test --features conformance`. The conclusion is the JSON report — deterministic,
reproducible by anyone. See `reference_vectors_anthropic()` (and `_openai` / `_gemini` /
`_responses`) for worked examples in four real wire shapes.

---

## 3. What the suite checks (and the two gate families)

- **codec-faithfulness** (round-trip / orphan-toolmsg #20 / interruption #19 / truncated-args
  #21): PASS = your codec behaved faithfully on a (possibly adversarial) input — it surfaced the
  planted defect instead of dropping it.
- **traffic-conformance** (model-identity #16 / face-purity #17): PASS = the response traffic is
  clean (no silent model reroute, no foreign usage fields).

A FAIL anywhere ⟹ a conformance issue in the codec or its traffic. Findings (#16/#17) are
orthogonal to LossObligation: a finding records an *event*; a LossObligation records *information
loss*. Don't conflate them.

---

## 4. The ⊥ floor (sanity anchor)

`BottomCodec` is the 0-anchor: it folds everything to the text floor, emitting one typed loss per
extension generator. It is intentionally NOT round-trip conformant — it is the maximal-loss
baseline. If your codec ever loses *more* than ⊥ on the same input, something is wrong.

---

## 5. Checklist before you claim "compatible"

- [ ] `up` and `down` implemented; kernel K round-trips losslessly (RoundTrip vector PASS).
- [ ] Every unrepresentable feature emits a typed loss (no silent drop) — both directions.
- [ ] Family prefixes correct (`behav.` / `bill.`).
- [ ] Vectors written in your wire shape; `run_conformance(...).passed()` is true.
- [ ] No raw content stored anywhere (hash + tokens only).
