# Behavioural-family findings — codex-cli → CC-Switch → 4 providers / 6 tiers

**Body:** codex-cli 0.144.6  **Bridge:** CC-Switch 3.17.0 (Local Routing, OpenAI-Chat anchor)
**Compute tiers (6):** deepseek-v4-pro · deepseek-v4-flash · kimi-k3 · kimi-k2.6 · glm-5.2 · mimo-v2.5-pro

Every verdict below is produced by the **agent-comm** library's own functions — no hand-written
"what counts as a loss" logic. The behavioural family (`≈_behav`) is measured across three layers,
shallow to deep, following the paper's framing that a syntactic neutral IR is near-trivial and the
real question is semantic faithfulness.

---

## Layer 1 — Structure (does the message shape survive the bridge?)

Tool: `translate_diff` — uses the paper's `translate("responses","openai", face1)` as the
**reference translation**, then compares its normal form against the bridge's actual output; both
sides lifted through the same OpenAI codec so the comparison is apples-to-apples.

Legend: ✗ = defect · — = identical to the reference · ⚠ = reasoning-only diff (codec caveat, §L3)

| Scenario | ds-flash | ds-pro | kimi-k3 | kimi-k2.6 | glm-5.2 | mimo-pro | Family (paper) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| **S06 mid-conversation `system`** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **role_collapse** |
| **S10 two-phase tool chain** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **translation** |
| S01 reasoning echo | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ | (reasoning, see §L3) |
| S03 interleave order | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ | (reasoning, see §L3) |
| S02/S04/S07/S09/S11/S12/S16/S17/S19/S20 | — | — | — | — | — | — | identical to reference |

**Two real defects, both 6/6 across all tiers — i.e. the bridge's doing, independent of which
model you route to.**

**S06** — a `system` instruction placed *mid-conversation* is moved to the front:
```
reference: User[text] | Assistant[text] | System[text] | User[text]
actual   : System[text] | User[text] | Assistant[text] | User[text]   <- system pulled to front
```
Effect: the rule's activation timing is rewritten — "in force from here on" becomes "in force from
the start". Harmless when the rule is time-invariant; wrong when it isn't (e.g. "switch tone now").

**S10** — two independent tool-call turns are merged into one assistant turn (reasoning folded in):
```
reference: User | Assistant[toolcall] | Assistant[toolcall] | Tool | Tool | User
actual   : User | Assistant[thinking+toolcall+toolcall] | Tool | Tool | User   <- turns merged, count -1
```
Effect: the turn boundary of the history is altered — any downstream logic that assumes
"one call per turn" will miscount steps.

---

## Layer 2 — Envelope (model identity + usage-field purity)

Tools: `check_model_identity` (#16) and `check_face_purity` (#17), both from the library. 181 cells.

**#16 model identity:** reroute = **0** across all 181 cells — no silent model swapping.

**#17 usage-field purity** (✗ = usage carries fields outside the face's native set):

| Scenario tally | ds-flash | ds-pro | kimi-k3 | kimi-k2.6 | glm-5.2 | mimo-pro |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| impure of 14 protocol scenarios | 14/14 | 14/14 | 10/14 | 12/14 | 0/14 | 0/14 |

Leaked fields: DeepSeek carries `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens`; Kimi carries
`cached_tokens`. GLM / MiMo are clean. **These leaked fields are the billing story** — see the
companion `billing_findings.md`; #17 flags them here as a behavioural boundary break, the billing
consequence is analysed there.

---

## Layer 3 — Semantics (does a downstream consumer actually care?)

Tool: `anchor_witness` — asks a real consumer (DeepSeek, temperature 0) for a response to each side
of a same-normal-form pair, then estimates ε per the paper's definition (only defined over same-nf
pairs; the estimator skips the rest).

- **ε ≈ 0** on plain-text / tool-cycle scenarios. The observed mixed ε (0.19) is *below* the model's
  own A/A non-determinism baseline (0.21, same request twice) — i.e. codec translation introduces
  **no measurable semantic loss**; the residual is the model's own stochasticity.
- **Reasoning blind spot, closed.** The ⚠ scenarios above are reasoning-placement differences
  (Anthropic `block` vs DeepSeek-style `inline_string`). The library's SPEC classifies
  `block ↔ inline_string` as a **FormalNormalize** (semantically equal, normalizable); the codec's
  missing normalization was implemented, so these now normalize to one form and enter ε measurement.
  Only placements touching `tool_call_kwargs` remain a true, non-normalizable loss (per the SPEC's
  §7.3 judgement table) — kept distinguishable on purpose.

---

## Honesty boundaries

- One bridge (CC-Switch), one anchor (OpenAI-Chat). Other bridges not in this drop.
- The ⚠ scenarios' reasoning caveat: a second codec asymmetry (the Responses codec treats *any*
  reasoning item as server-managed, plaintext or not) means the reasoning layer is not yet fully
  judged for CC-Switch specifically; the placement normalization above is verified on the library's
  own reasoning samples, not yet re-run through the CC-Switch reasoning path.
- ε is measured on the library's own semantic samples with a single downstream consumer; the
  CC-Switch **link's** ε (feeding CC-Switch's captured pairs to a real consumer) is not yet run.
- Verdicts are field-/structure-level (counts and normal-form equality), not byte-level.
