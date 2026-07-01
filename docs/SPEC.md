# agent-comm Protocol Specification v0.2.0

> **Status: Stable.** The kernel K wire format (Text/ToolCall/ToolResult) and extension generators (Thinking/Media/Video) are frozen. Future versions will maintain backward compatibility.
>
> **Language**: This document uses technical English terms for precision. Chinese translations of key terms are noted in parentheses on first use.

---

## 1. Introduction

### 1.1 Background

Every LLM provider speaks a different wire format:

| Provider | Tool call format | Tool result format | Reasoning |
|---|---|---|---|
| Anthropic / Claude | `tool_use` block | `tool_result` block inside user role | `thinking` block |
| OpenAI / GPT | `tool_calls` array on assistant message | standalone `tool` role message | `reasoning_content` field |
| Google / Gemini | `functionCall` part | `functionResponse` part | `thought` part |
| DeepSeek etc. | OpenAI-compatible `tool_calls` | OpenAI-compatible `tool` role | `reasoning_content` field |

These differences mean that a conversation that makes sense in one provider's format must be **translated** before it can be used with another provider. Traditional approaches either:

- **Pairwise converters** — An O(n²) problem: every new provider requires converters to and from every existing provider. Tools like `tool_call` arguments get silently dropped or coerced (e.g. a truncated JSON string becomes `{}`).
- **Black-box proxies** — Services that translate between formats without telling the caller what was lost. There is no audit trail.

### 1.2 What this protocol solves

agent-comm defines a **neutral conversation IR** (intermediate representation) and a set of **codec** interfaces. Every provider converts **to** and **from** this IR — never pairwise between providers:

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│ Anthropic │────▶│              │◀────│  OpenAI  │
└──────────┘     │   Neutral    │     └──────────┘
┌──────────┐     │   IR +       │     ┌──────────┐
│  Gemini  │────▶│  Normal Form │◀────│Responses │
└──────────┘     └──────────────┘     └──────────┘
```

Key guarantees:

- **Every conversion that loses information records a typed LossObligation** — nothing is silently dropped.
- **Two conversations are semantically equal iff their normal forms are equal** — this is decidable.
- **All encode/decode operations are fail-closed** — unauthorized access is rejected, never silently bypassed.

### 1.3 Scope and boundaries

| In scope | Out of scope |
|---|---|
| Conversation structure (turns, roles, content blocks) | Network transport (HTTP, WebSocket, SSE) |
| Tool calls and tool results | API key management or secrets |
| Reasoning / thinking content | Prompt caching implementation |
| Multi-modal media (images, video) | Model selection or routing |
| Cross-vendor translation fidelity | Runtime streaming |
| Loss accounting (behavioral + billing) | User identity or authentication beyond Principal |

---

## 2. Terminology

| Term | Definition |
|---|---|
| **Kernel K** | The frozen core set of content generators: `Text`, `ToolCall`, `ToolResult`. Every codec must support these. |
| **Extension generator** | Optional but frozen content types: `Thinking`, `Media`, `Video`. Codecs that cannot express them record a loss rather than failing. |
| **Conversation** | An ordered sequence of `Turn` values, each consisting of a `Role` and a list of `Content`. |
| **Normal form** `nf(·)` | A deterministic normalization pipeline (R5→R1→R6→R3→R2). The normal form is unique: `nf(nf(x)) == nf(x)`. |
| **LossObligation** | A typed record of a feature that a codec could not carry. Two-sided: both `up` (native→IR) and `down` (IR→native) may produce losses. |
| **Envelope** | Request-level or response-level metadata (model name, token usage, fingerprints) that lives outside the IR kernel. |
| **Fail-closed** | An operation that refuses to proceed when the caller is not authorized, rather than silently degrading. |

---

## 3. Protocol Architecture

The protocol operates at four layers, numbered L1 (lowest) to L4 (highest):

```
L4: Semantic IR      ─  Conversation, Content, Turn, Role
L3: Envelope         ─  model, max_tokens, tools, usage, fingerprint
L2: Native wire      ─  Provider-specific JSON (Claude messages[], GPT messages[], etc.)
L1: Transport        ─  HTTP, WebSocket, SSE (outside spec scope)
```

**L4 is the protocol.** L3 is a carrier for request/response metadata. L2 is vendor-accidental and is converted to L4 by codecs. L1 is explicitly out of scope.

---

## 4. Core Data Types

### 4.1 Role

A `Turn` carries a role. The protocol defines four:

```
Role ::= System | User | Assistant | Tool
```

Vendors have various spellings for the same concept. The canonical mapping is:

| Canonical Role | Vendor aliases |
|---|---|
| `System` | `system`, `developer` |
| `User` | `user`, `human` |
| `Assistant` | `assistant`, `ai`, `model` |
| `Tool` | `tool`, `function` |

### 4.2 Content generators

A `Content` value is one of the following variants. The first three form the **kernel K** (frozen, all codecs must support them). The remaining three are **extension generators** (frozen, codecs may record a typed loss if they cannot express them).

#### 4.2.1 Kernel K (frozen, mandatory)

```
Content::Text { text: String, cache_control: Option<CacheControl> }
```

A plain-text block. `cache_control` carries prompt-cache breakpoint metadata and belongs to the billing equivalence class (≈_bill, i.e. it affects cost accounting, not behavior). A codec that cannot express `cache_control` must emit `billing.cache_directive_lost`.

```
Content::ToolCall { id: CallId, name: ToolName, args: Value }
```

A tool/function invocation. `args` is a JSON object whose key ordering is vendor-accidental; the normal form sorts keys deterministically (R2). A truncated or unparseable `args` string must **not** be silently coerced to `{}` — the codec must emit `behavior.truncated_args`.

```
Content::ToolResult { ref_id: CallId, payload: Vec<Content>, cache_control: Option<CacheControl> }
```

A tool result. `ref_id` must reference a preceding `ToolCall.id` (validated by `Conversation::validate`). N1 constraint: `payload` must not contain `ToolCall`, `Thinking`, or `Video`; nesting depth ≤ 2.

#### 4.2.2 Extension generators (frozen, may incur loss)

```
Content::Thinking { text: String, sig: Option<String>, placement: Placement }
```

Reasoning / chain-of-thought. `placement` records where the reasoning sits in the native wire (it belongs to the **behavior** equivalence class — changing placement can change runtime behavior, e.g. Letta's tool dispatch). Collapsing `placement` from `ToolCallKwargs` incurs `behavior.placement_collapsed`.

```
Content::Media { mime: String, data: String }
```

Inline base64 media (image/* or application/pdf). Permitted inside `ToolResult.payload`.

```
Content::Video { source: VideoSource, mime: String, duration_seconds: Option<u64> }
```

Video content with either URL or base64 source. N1: **not** permitted inside `ToolResult.payload`.

### 4.3 Placement

Reasoning placement (G1 ruling):

| Placement | Native position | Example vendors |
|---|---|---|
| `Block` | Dedicated reasoning block | Anthropic `thinking`, Gemini `thought` |
| `ToolCallKwargs` | Embedded in tool call arguments | Letta `put_inner_thoughts_in_kwargs` |
| `InlineString` | Inline string field | DeepSeek `reasoning_content` |

### 4.4 Normal Form

The normal form `nf(c)` of a conversation `c` is computed by applying the following pipeline **in order**:

| Step | Name | Effect |
|---|---|---|
| **R5** | Drop empty text | Remove empty `Text` blocks; recurse into `ToolResult.payload` |
| **R1** | Merge adjacent text | Concatenate adjacent `Text` blocks if their `cache_control` is identical |
| **R6** | Tool role canon | Hoist each `ToolResult` into its own `Role::Tool` turn, preserving order |
| **R3** | ID canon | Rename `ToolCall.id` values to `call_0, call_1, ...` in first-appearance order |
| **R2** | Args key sort | Recursively sort `ToolCall.args` object keys (BTreeMap rebuild) |

Properties:

- **Idempotent**: `nf(nf(c)) == nf(c)`
- **Semantic equality**: `c1.semantic_eq(c2)` iff `nf(c1) == nf(c2)`
- **Deterministic**: same input always produces the same output

---

## 5. Loss Accounting

### 5.1 LossObligation

When a codec cannot carry a feature from one representation to another, it records a typed obligation. Losses are **two-sided**: both `up` (native → IR) and `down` (IR → native) may produce losses.

```
LossObligation {
    provider: String,         // e.g. "openai", "gemini"
    dropped_kind: String,    // typed loss family
    turn_index: usize,       // where the loss occurred
    recoverable: bool,       // can the feature be recovered on return trip?
    note: String,            // human-readable description
    detail: Option<BTreeMap<String, String>>,  // structured typed payload
}
```

### 5.2 Two-account separation

Losses are split into two families, tracked independently:

| Family prefix | Account | Examples |
|---|---|---|
| `behavior.*` | **Behavior account** (≈ the customer experience) | `behavior.truncated_args`, `behavior.placement_collapsed` |
| `billing.*` | **Billing account** (≈ the cost/usage side) | `billing.cache_directive_lost` |

A `behavior.*` loss means the consumer may observe different behavior. A `billing.*` loss means billing accuracy may be affected — but the behavior is intact.

### 5.3 R-3 rule (never silent drop)

> **If your wire cannot express a kernel feature, you must record a LossObligation. You must not silently drop it, coerce it to a default value, or fabricate a placeholder.**

---

## 6. Envelope

### 6.1 Request envelope

Request-level fields that do **not** enter the IR kernel:

| Field | Description |
|---|---|
| `model` | Model identifier (e.g. `claude-3-5-sonnet`) |
| `max_tokens` | Maximum output tokens |
| `temperature` / `top_p` | Sampling parameters |
| `tools` | Tool/function definitions |

The `split_envelope` function separates a native request into (conversation IR, envelope). The `apply_envelope` function reattaches envelope fields after `down`.

### 6.2 Request fingerprint (G8)

A canonical hash of the native request body:

```
request_fingerprint = blake3(canonical_sort(native_body))
```

Object keys are sorted deterministically before hashing. Same body under different JSON serializations → same fingerprint.

### 6.3 Response envelope

Response-side metadata:

| Field | Description |
|---|---|
| `echoed_model` | Provider's self-reported model id |
| `usage` | Token accounting (input_tokens, output_tokens, etc.) |
| `stop_reason` | Termination reason |
| `response_fingerprint` | Canonical hash of the response body |

### 6.4 Cache freshness (G9)

| Value | Meaning |
|---|---|
| `Fresh` | Provider confirmed cached response still matches current ground truth |
| `Stale` | Provider flagged cache expired but served it anyway |
| `Unknown` | No freshness signal (conservative default) |

---

## 7. Conformance

### 7.1 Codec contract

Every `ProviderCodec` implementation must satisfy:

```
check_round_trip(codec, native)  ⟹  nf(up(codec, native)) == nf(up(codec, down(codec, nf(up(codec, native)))))
```

In words: converting a native message through the IR and back, then normalizing,
must yield the same normal form as converting it once and normalizing.

### 7.2 Conformance gates

A conformant implementation passes all gates in the suite:

| Gate | What it checks |
|---|---|
| **Round-trip** | Lossless round-trip on kernel K |
| **Orphan tool_result (#20)** | Tool results with no matching call are surfaced, not dropped |
| **Abandoned tool_call (#19)** | Interrupted mid-conversation calls are surfaced, not fabricated |
| **Truncated args (#21)** | Truncated JSON arguments emit `behavior.truncated_args`, not silent `{}` |
| **Model identity (#16)** | `echoed_model ≠ requested_model` → Reroute finding |
| **Face purity (#17)** | Usage fields are native to the claimed provider face |

### 7.3 Bottom codec (⊥)

The bottom codec is the **maximal-loss baseline**: it expresses only `Text` and drops everything else with a typed loss. It is the reference floor: any codec that loses *more* than ⊥ has a bug.

---

## 8. Security

### 8.1 Fail-closed operations

Every `encode` and `decode` call requires an authorized `Principal`:

```
encode(conv, codec, who: &Principal, gate: &dyn IdentityGate) -> Result<...>
decode(native, codec, who: &Principal, gate: &dyn IdentityGate) -> Result<...>
```

If the gate does not authorize the principal, the operation returns `CommError::Unauthorized`. There is no bypass path.

### 8.2 Data handling

- No raw content is stored — only `response_fingerprint` (a hash) and aggregated usage counts.
- The `anchor_witness` example collects only `blake3(response_text)` from real providers; raw response text never leaves the adapter scope.

---

## 9. Versioning

| Version | Status | Changes |
|---|---|---|
| 0.1.0 | Superseded | Initial protocol Step1 + Step2 tooling |
| 0.2.0 | **Current stable** | Kernel K frozen, extension generators frozen, envelope dual-end, LossObligation v0.3, conformance suite finalized |

Future versions (≥ 0.3.0) will maintain backward compatibility for the kernel K wire format. Extension generators may gain new variants but existing ones will not be removed or redefined.

---

## 10. References

- `src/lib.rs` — Reference implementation of the IR, normalize, encode/decode, translate
- `src/protocol.rs` — Principal and IdentityGate contract
- `src/codecs/` — Reference codecs (Anthropic, OpenAI, Gemini, Responses, Bottom)
- `src/conformance.rs` — Conformance suite runner and reference vectors
- `tests/conformance.rs` — 100+ conformance tests
- `CONTRIBUTING.md` — How to add a new codec
- `CODEC_BUILDER_GUIDE.md` — Step-by-step guide for codec authors

---

> Spec version 0.2.0. License: MIT OR Apache-2.0.
