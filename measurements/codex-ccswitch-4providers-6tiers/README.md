# agent-comm — CC-Switch bridge measurements (behavioural + billing)

> **This test:** `Codex (codex-cli 0.144.6) × CC-Switch bridge × 4 providers / 6 tiers`,
> measured across the **behavioural family** and the **billing family** separately.
> (中文备注：本次测试 = Codex-CCSwitch-四家六档 × 行为族/计费族)

What a real routing bridge (**CC-Switch 3.17.0**, OpenAI-Chat anchor) does to conversations and to
**billing metadata** when a coding agent (**codex-cli 0.144.6**) reaches six model tiers across four
providers through it.

Every verdict is produced by the **agent-comm** neutral-IR library's own functions — no hand-written
"what counts as a loss" logic. Where a mathematical judgement was required, it went through an
independent formal adjudication.

> Companion library: **agent-comm** (neutral conversation IR + two-sided loss accounting).
> The tools in `tools/` are `examples/` of that library.

## Two headlines

**Behavioural** (`reports/behavioural_findings.md`) — measured with `translate_diff` (the paper's
`translate()` as reference) + the `#16/#17/#19/#20` gates:
- `S06` mid-conversation `system` is **moved to the front** — 6/6 tiers (role_collapse).
- `S10` two independent tool-call turns are **merged into one** — 6/6 tiers (translation).
- Semantics: codec translation shows **no measurable loss** (ε below the model's own A/A baseline);
  the reasoning-placement blind spot was closed by implementing the SPEC's `block ↔ inline_string`
  normalization.

**Billing** (`reports/billing_findings.md`) — the costly one:
- The bridge "cleans" `usage` into `input/output/total` and **deletes the cache-billing fields**
  (`prompt_cache_hit/miss_tokens`, `cached_tokens`).
- DeepSeek's cache-hit vs cache-miss gap is **up to 120×** (official pricing).
- **ρ_drop ≈ 0.48** of calls lose the ability to tell whether they hit cache.
- The purity gate reports the post-bridge account as **"pure" (0 violations) — precisely because the
  billing info was cleanly deleted.** Ruled a valid `≈_bill` loss and the first field-realistic
  instance of the paper's Corollary 5.2 (faithful to one family, collapsed on an incomparable one).

## Layout

```
reports/behavioural_findings.md   structure / envelope / semantic layers × 6 tiers
reports/billing_findings.md       cache-field deletion, price gaps, ρ_drop, dual-gate case
tools/                            agent-comm examples used as the judges (Rust; keys from env only)
```

## Reproduce

The `tools/*.rs` are `examples/` of the agent-comm crate. From an agent-comm checkout:

```bash
cargo run -p agent-comm --example translate_diff   -- <scenarios_root>
cargo run -p agent-comm --example gate_scan        -- <captures_root> openai
cargo run -p agent-comm --example gate_bridge_resp -- <scenarios_root> responses
DEEPSEEK_API_KEY=... cargo run -p agent-comm --example anchor_witness   # ε (real consumer)
```

## Honesty boundaries

- One bridge (CC-Switch), one anchor (OpenAI-Chat).
- `ρ_drop` (per-call collapse rate) and paired `ε_bill` are **different statistics** — reported
  separately, never mixed.
- Billing loss = **audit blindness**, not "overpaying the vendor"; the client can no longer see,
  verify, or optimise the biggest cost lever (caching).
- GLM / MiMo returned no cache fields → not applicable, recorded as such (not "clean").
- Raw captures, provider-key backups, and runtime dirs are excluded (see `.gitignore`); this drop is
  the findings, the tools, and the reproduce steps only.

## Licence

Reports: CC-BY-4.0 · Tool sources: Apache-2.0 (same as agent-comm).
