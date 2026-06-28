# agent-comm

A neutral conversation IR: a single `Conversation` type that is a **colimit over a small kernel**
of content generators. Every provider format converts to/from this IR via a `ProviderCodec` —
never pairwise between providers.

## What's in v0

- **Kernel K**: `Text` / `ToolCall` / `ToolResult` (wire frozen).
- **Extension generators** (wire EXPERIMENTAL, not frozen): `Thinking` / `Media` / `Video`.
- **Codecs**: `AnthropicCodec` / `OpenAiCodec` / `GeminiCodec` / `ResponsesCodec`.
- **Normal form** `Conversation::normalize` — the unique nf(·) computed as the pipeline
  **R5 → R1 → R6 → R3 → R2**:
  - R5 drop empty text · R1 merge adjacent text · R6 fold `tool_result` into `Tool` turns
    (vendor-accidental role placement canonicalized) · R3 canonical `call_0, call_1, …` ids ·
    R2 sort `ToolCall.args` object keys.
  - Idempotent: `nf(nf(x)) == nf(x)`. Two conversations are semantically equal iff their normal
    forms are equal (`Conversation::semantic_eq`).
- **Validation** `Conversation::validate` — N1 (`tool_result.payload` may not contain
  `ToolCall` / `Thinking` / `Video`; nesting depth ≤ 2), tool-link closure (every
  `tool_result.ref` points to a preceding `tool_call.id`), and `ToolCall` id uniqueness.
- **Loss accounting** — two-sided `LossObligation`: both `up` (native → IR) and `down`
  (IR → native) may report loss; conversions that cannot carry a feature record a typed
  obligation rather than silently dropping it.
- **Cross-vendor translation** `translate(from, to, native)` =
  `down_to ∘ up_from` (the decomposition theorem in code form). Replaces hand-written
  pairwise `X_to_Y` converters.
- **Request envelope** `split_envelope` / `apply_envelope` — request-level fields
  (`model` / `max_tokens` / `tools` / …) never enter the IR kernel; the envelope layer
  preserves them.
- **Codec registry** `codec_for(provider)` — alias normalization
  (`claude → anthropic` / `chat → openai` / `codex → responses` / …).
- **Round-trip conformance** `check_round_trip(codec, native)` — verifies
  `up → normalize → down → up → normalize` yields the same normal form.
- **Identity** `encode` / `decode` are **fail-closed**: each requires an authorized
  `Principal` via `agent_protocol::IdentityGate`; unauthorized → `CommError::Unauthorized`.

## Extending

Adding a generator = adding a `Content` variant = adding a leg to the colimit; existing
variants and their wire format do not change. Adding a provider = adding a `ProviderCodec`
impl + a `codec_for` arm; other codecs are not touched (no privileged anchor — providers
are peers).

## Dependencies

`agent-protocol` (the standard contract), `serde`, `serde_json`. No async runtime,
no networking, no codec adapters between providers.

## License

`MIT OR Apache-2.0`.

## Math

TBA — companion papers in preparation.
