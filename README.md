# agent-comm

[![中文](https://img.shields.io/badge/中文-README-blue)](README.zh-CN.md)

**A neutral format for multi-provider LLM conversations. Switch between Claude, GPT, Gemini, DeepSeek — without format lock-in, without silently losing data.**

A single `Conversation` type that is a **colimit over a small kernel K** of content generators. Every provider format converts to/from this IR via a `ProviderCodec` — **never pairwise between providers**.

---

## The problem

Every LLM provider speaks a different dialect. Anthropic uses `tool_use` blocks, OpenAI uses `tool_calls`, Gemini uses `functionCall`. If you want to support multiple providers, you're forced to either:

> **Vendor-lock in, or write N² pairwise converters and never know what got silently lost.**

Traditional translation proxies convert formats but don't tell you — that `thinking` block got dropped, that truncated tool call argument became `{}` without anyone noticing.

**agent-comm solves this with a single neutral IR:**

```
Not:  Anthropic ──→ OpenAI          (n² converters, cost doubles per provider)
      Gemini   ──→ DeepSeek
      ...

But:  Anthropic ──┐
      OpenAI    ──┼──→ Neutral IR ──→ any provider
      Gemini    ──┘                   (2n converters, one per provider)
```

---

## Quick start

```bash
cargo add agent-comm

use agent_comm::translate;
let openai_output = translate("anthropic", "openai", &claude_json)?;
```

Zero async runtime, zero networking — pure data transformation. All encode/decode operations are **fail-closed**: unauthorized access returns an error, never a guess.

---

## What's in v0

| Capability | Description |
|---|---|
| **Kernel K** (wire frozen) | `Text` / `ToolCall` / `ToolResult` |
| **Extension generators** (wire frozen) | `Thinking` / `Media` / `Video` |
| **Reference codecs** | `AnthropicCodec` / `OpenAiCodec` / `GeminiCodec` / `ResponsesCodec` |
| **Normal form** | R5→R1→R6→R3→R2 pipeline. Idempotent. Two conversations from different providers are semantically equal iff their normal forms are equal |
| **Loss accounting** | Two-sided `LossObligation` (up and down). Every lost feature is typed — never silently dropped |
| **Cross-vendor translation** | `translate()` = `down_to ∘ up_from`. Eliminates all handwritten `X_to_Y` converters |
| **Request envelope** | `split_envelope` / `apply_envelope` — model, tokens, fingerprint stay outside the IR |
| **Codec registry** | `codec_for()` — alias normalization (`claude → anthropic` / `chat → openai` / `codex → responses`) |
| **Round-trip conformance** | `check_round_trip()` — verifies `up → normalize → down → up → normalize` yields the same normal form |
| **Conformance suite** | `cargo test --features conformance` — 7 gates. PASS = "agent-comm compatible" stamp |
| **Metrics & Φ coords** | 3-axis complexity (Genesis / Stratification / Reification) + empirical ε estimate |

---

## Design principles

> These aren't slogans — every codec MUST follow them. The conformance suite enforces them.

### Never drop silently (R-3)

If your wire cannot carry a feature, **you must say so — you cannot silently digest it**.

A truncated JSON args string must NOT become `{}` — it becomes `LossObligation { dropped_kind: "behav.truncated_args" }`. Reasonix's `closeTruncatedJSON → "{}"` is the textbook counterexample: the caller never knows the args were lost.

### Fail closed

When in doubt, reject rather than guess. Bad identity → no encode/decode. Malformed input → `CodecError::Malformed`. No "let me figure out what you meant."

### Two-sided accounting (Cor5.2)

Behavioral loss and billing loss are tracked separately:

| Account | Prefix | Examples |
|---|---|---|
| ≈_behav (behavior) | `behav.*` | `behav.placement_collapsed`, `behav.truncated_args` |
| ≈_bill (billing) | `bill.*` | `bill.cache_directive_lost` |

A lost cache breakpoint (bill loss) does not mean your conversation content is wrong. Two accounts, independently auditable.

### Vendor-neutral

In `codec_for()`, Anthropic and Kimi, Claude and DeepSeek — they are structurally equal. No "Anthropic is the core, others are add-ons." Vendor neutrality means the same judgment function applies to every provider.

---

## Extending

### Add your own provider

Copy the template → fill 5 TODOs → register → run conformance → open a PR.

```bash
cp docs/codec_template.rs src/codecs/my_provider.rs
# Fill 5 TODOs, register in codec_for()
cargo test --features conformance   # PASS = stamp
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODEC_BUILDER_GUIDE.md](docs/CODEC_BUILDER_GUIDE.md) for the full guide.

### Add a generator

Add a `Content` variant = add a leg to the colimit. Existing variants and their wire format don't change.

### Report bugs / submit PRs

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Dependencies

`serde`, `serde_json`, `blake3`. Zero external protocol dependencies. No async runtime, no networking.

---

## License

MIT OR Apache-2.0.
