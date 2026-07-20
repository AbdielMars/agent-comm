# Billing-family findings — the bridge quietly rewrites your bill

**Body:** codex-cli 0.144.6  **Bridge:** CC-Switch 3.17.0 (OpenAI-Chat anchor)
**Compute tiers:** deepseek-v4-pro/flash · kimi-k3/k2.6 · glm-5.2 · mimo-v2.5-pro

The billing family (`≈_bill`) and the behavioural family are, by the paper's Corollary 5.2,
**incomparable** — no single core is faithful to both. So they are accounted separately. This is
the family that costs money.

---

## The headline

A routing bridge "cleans" the upstream `usage` object into a standard `input/output/total` shape —
and in doing so **deletes the cache-billing fields**. Measured directly, offline, on captured
before/after pairs (no extra API calls):

```
Upstream (DeepSeek, real usage):
  prompt_tokens:10, completion_tokens:43, total:53,
  prompt_cache_hit_tokens:0, prompt_cache_miss_tokens:10, cached_tokens:0

Client-facing (CC-Switch re-encoded usage):
  input_tokens:10, output_tokens:43, total:53, output_tokens_details:{...}
  -> prompt_cache_hit/miss_tokens and cached_tokens are GONE
```

The three fields that determine cache pricing are removed. Token totals are preserved (53 = 53);
the *cache breakdown* — the one thing that moves cost — is not.

---

## Why it matters: the price gap (official rate cards, checked directly)

| Tier | cache-hit /1M | cache-miss /1M | gap | source |
|---|---|---|---|---|
| deepseek-v4-pro | $0.003625 | $0.435 | **120×** | api-docs.deepseek.com/quick_start/pricing |
| deepseek-v4-flash | $0.0028 | $0.14 | **50×** | (same) |
| kimi-k3 | ¥2.00 | ¥20.00 | **10×** | platform.kimi.com/docs/pricing/chat-k3 |
| glm / mimo | — | — | — | no cache fields observed → not applicable |

A cache hit costs up to **120× less** than a miss. Delete the hit/miss breakdown and the client can
no longer tell which one happened.

---

## How much is lost — the corrected statistic

Scanned all captured upstream responses offline: **57 of 120 calls carried a real cache hit** (up to
2432 hit tokens).

- **ρ_drop = 57 / 120 ≈ 0.48** — the fraction of calls where the client-facing usage can no longer
  reflect the true cache breakdown (an observation-collapse rate).
- This is deliberately **not** called ε. The paper's ε is a *paired* statistic over
  representation-equal messages; ρ_drop is a *per-call* collapse frequency. They are different
  quantities and are reported separately, never mixed.
- Audit-overestimate formula (client, seeing only post-bridge usage, must price all input at the
  miss rate): `Σ hit_tokens × (miss_price − hit_price)`. On this short test corpus the absolute
  figure is tiny (small token counts); in production it scales linearly with `hit_tokens × gap`
  (10–120×).

---

## The structural point: existing purity gates are blind to this

Running the #17 face-purity gate on the **client-facing** response (as a `responses` face):

```
post-bridge #17 FaceImpurity = 0   (all 96 client-facing responses "pure")
pre-bridge  #17 FaceImpurity = 104 (DeepSeek/Kimi hit/miss present, non-native to openai)
```

The post-bridge account reads **perfectly pure — precisely because the billing fields were cleanly
deleted.** A purity gate checks that usage contains nothing it shouldn't; it is structurally blind
to usage *missing* something it should have. "Cleaner" here means "the cost information was removed."

This is the case for a **dual gate**: purity checks `usage ⊄ native(face)` (nothing extra); the
missing direction is `usage shrank relative to upstream on a price-bearing field`. The two directions
together close usage-layer faithfulness — mirroring the paper's soundness/separation pair at the
field level. **An independent mathematical adjudication confirmed** this qualifies as a genuine
`≈_bill` loss (the client's observable billing partition strictly coarsens), and that it is the first
field-realistic instance of the paper's Corollary 5.2: a bridge faithful to the behavioural family
while collapsing an incomparable billing family. The severity criterion is zero-discretion: only
fields that are **arguments of the official price function** (hit/miss/cached) count as a billing
loss; other usage fields do not.

---

## Bottom line

- The loss is **audit blindness, not overpaying the vendor** — the provider still bills the real
  cache hit. The client can no longer *see, verify, or optimise* the single biggest cost lever.
- Published rate cards are the sticker price; the bill is what you pay — and **once it passes through
  a bridge, the number on the bill isn't even the number on the bill.**

## Honesty boundaries

- One bridge (CC-Switch), DeepSeek/Kimi only; GLM/MiMo returned no cache fields → not applicable,
  recorded as such, not "clean".
- ρ_drop (per-call) and a paired ε_bill are different statistics; the paired estimator needs enough
  same-view buckets, and where a bucket has no pair it is reported as "undefined", not forced.
- Rate cards captured from official pricing pages on the date noted; enterprise/negotiated rates are
  out of scope (and, as others have noted, differ from the card anyway — which only sharpens the
  point that you must be able to *read* the cache breakdown to reason about real cost).
