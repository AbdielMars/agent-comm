# Anchor Witness RESULTS v2

## Mode
[MIXED@anthropic=MOCK, openai=MOCK, gemini=MOCK, deepseek=REAL]

## Environment
- rustc: rustc (compile-time unknown)
- agent-comm: 0.1.0
- c-meter: 0.1.0
- Generated: build-time

## Sample Set
- Total pairs: 50
- Distribution:
  - multimodal: 12 (nf_match 12, nf_mismatch 0)
  - plain_text: 14 (nf_match 14, nf_mismatch 0)
  - thinking: 12 (nf_match 12, nf_mismatch 0)
  - tool_cycle: 12 (nf_match 12, nf_mismatch 0)

## Codecs Tested
- anthropic (MOCK)
- openai (MOCK)
- gemini (MOCK)
- deepseek (REAL)

## Results
- ε (sup over observers): 0.2631578947368421
- nf_match / nf_mismatch / upstream_skipped: 50/0/12
- target observer: `deepseek` (codec: `openai`)

## Reproduce
```bash
cargo run --example anchor_witness
```

## Notes
- Mock providers hash request bytes; ε under a fully-mock fleet is 0 by
  construction (same nf → identical request normalization → identical hash).
- For any provider in REAL mode, ε > 0 is expected due to LLM stochasticity;
  the magnitude indicates how often same-nf inputs yield divergent outputs.
- Privacy: only `blake3(response_text)` is collected from real providers;
  raw response content is discarded inside the adapter.
