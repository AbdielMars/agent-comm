//! anchor_witness — empirical ε measurement against real or mock consumers.
//!
//! For each sample pair `(provider_a, native_a) ↔ (provider_b, native_b)` we:
//!   1. Lift both natives through their codecs into the IR and normalize.
//!   2. Skip pairs whose normal forms differ (precondition for ε).
//!   3. Ask the per-provider upstream adapter for a `blake3(response)` signature for each
//!      member of the pair (the adapter is either a deterministic mock or a real HTTP
//!      call, picked by `{PROVIDER}_API_KEY` env-var presence).
//!   4. Estimate ε = `sup_o Pr_{same nf}[response_o(m1) != response_o(m2)]` via
//!      `agent_comm::metrics::epsilon_estimate`.
//!
//! Privacy: only blake3 hashes flow back from the adapter. Raw response text never leaves
//! the per-provider `call`. The samples are dumped to `examples/anchor_samples.json` for
//! reproducibility; RESULTS.md persists only aggregate statistics.

use agent_comm::codecs::{AnthropicCodec, GeminiCodec, OpenAiCodec, ResponsesCodec};
use agent_comm::metrics::{epsilon_estimate, ResponseVector};
use agent_comm::{translate, Conversation, ProviderCodec};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::time::Duration;

const REQ_TIMEOUT: Duration = Duration::from_secs(30);

// ============================================================================
// Local Upstream trait (B route: not depending on the private c-meter crate)
// ============================================================================

#[async_trait]
trait Upstream: Send + Sync {
    #[allow(dead_code)] // surfaced for debugging / future logging; not used by main()
    fn provider_id(&self) -> &'static str;
    async fn call(&self, req: Value) -> Result<Vec<u8>>;
}

struct MockUpstream {
    provider: &'static str,
}

#[async_trait]
impl Upstream for MockUpstream {
    fn provider_id(&self) -> &'static str {
        self.provider
    }
    /// Mock = IR-nf-faithful by construction: lift the request through the provider's
    /// codec, normalize, then hash the normal form. Same nf → same hash → ε = 0 under a
    /// fully-mock fleet (consumer family that is faithful to the IR).
    async fn call(&self, req: Value) -> Result<Vec<u8>> {
        let conv = lift(self.provider, &req).context("mock: lift to IR")?;
        let bytes = serde_json::to_vec(&conv).context("mock: serialize nf")?;
        let mut h = blake3::Hasher::new();
        h.update(&bytes);
        Ok(h.finalize().as_bytes().to_vec())
    }
}

struct RealUpstream {
    provider: &'static str,
    base_url: &'static str,
    api_key_env: &'static str,
}

#[async_trait]
impl Upstream for RealUpstream {
    fn provider_id(&self) -> &'static str {
        self.provider
    }
    async fn call(&self, req: Value) -> Result<Vec<u8>> {
        let key = std::env::var(self.api_key_env)
            .map_err(|_| anyhow!("missing env var {}", self.api_key_env))?;
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let client = reqwest::Client::builder()
            .timeout(REQ_TIMEOUT)
            .build()
            .context("build client")?;
        for attempt in 0..2 {
            let resp = client.post(&url).bearer_auth(&key).json(&req).send().await;
            match resp {
                Ok(r) => {
                    let status = r.status();
                    let body = r.text().await.context("read body")?;
                    if status.is_success() {
                        // Hash only the reply content (`choices[0].message.content`),
                        // not the whole response body — bodies contain per-call unique
                        // fields like `id` and `created` (unix timestamp) that would
                        // make hash always differ. Privacy: the reply text itself is
                        // never exposed; only its blake3 leaves this scope.
                        let parsed: Value = serde_json::from_str(&body)
                            .unwrap_or_else(|_| Value::String(body.clone()));
                        let content = parsed
                            .pointer("/choices/0/message/content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let mut h = blake3::Hasher::new();
                        h.update(self.provider.as_bytes());
                        h.update(b"\0");
                        h.update(content.as_bytes());
                        return Ok(h.finalize().as_bytes().to_vec());
                    }
                    if status.is_server_error() && attempt == 0 {
                        continue;
                    }
                    return Err(anyhow!("HTTP {status}"));
                }
                Err(_) if attempt == 0 => continue,
                Err(e) => return Err(anyhow!("request failed: {e}")),
            }
        }
        Err(anyhow!("retries exhausted"))
    }
}

fn upstream_for(provider: &str) -> Option<Box<dyn Upstream>> {
    let preset = match provider {
        "anthropic" => RealUpstream {
            provider: "anthropic",
            base_url: "https://api.anthropic.com/v1",
            api_key_env: "ANTHROPIC_API_KEY",
        },
        "openai" => RealUpstream {
            provider: "openai",
            base_url: "https://api.openai.com/v1",
            api_key_env: "OPENAI_API_KEY",
        },
        "gemini" => RealUpstream {
            provider: "gemini",
            base_url: "https://generativelanguage.googleapis.com/v1beta",
            api_key_env: "GEMINI_API_KEY",
        },
        "deepseek" => RealUpstream {
            provider: "deepseek",
            base_url: "https://api.deepseek.com/v1",
            api_key_env: "DEEPSEEK_API_KEY",
        },
        _ => return None,
    };
    if std::env::var(preset.api_key_env)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
    {
        Some(Box::new(preset))
    } else {
        Some(Box::new(MockUpstream {
            provider: preset.provider,
        }))
    }
}

fn mode_label(provider: &str) -> &'static str {
    let env = format!("{}_API_KEY", provider.to_uppercase());
    if std::env::var(env)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
    {
        "REAL"
    } else {
        "MOCK"
    }
}

// ============================================================================
// Sample generation (4 kinds × 12 pairs each = 48 pairs)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SamplePair {
    label: String,
    kind: &'static str,
    provider_a: &'static str,
    native_a: Value,
    provider_b: &'static str,
    native_b: Value,
}

fn lift(provider: &str, native: &Value) -> Result<Conversation> {
    let codec: Box<dyn ProviderCodec> = match provider {
        "anthropic" => Box::new(AnthropicCodec),
        // DeepSeek's HTTP wire is OpenAI-compatible; route it through the OpenAI codec.
        "openai" | "deepseek" => Box::new(OpenAiCodec),
        "gemini" => Box::new(GeminiCodec),
        "responses" => Box::new(ResponsesCodec),
        other => return Err(anyhow!("unknown provider {other}")),
    };
    let (c, _) = codec
        .up(native)
        .map_err(|e| anyhow!("up({provider}): {e}"))?;
    Ok(c.normalize())
}

fn gen_plain_text() -> Vec<SamplePair> {
    let pairs = [
        ("hi", "hello"),
        ("good morning", "morning"),
        ("how are you", "I am fine"),
        ("thanks", "you are welcome"),
        ("ping", "pong"),
        ("yes", "no"),
        ("ok", "got it"),
        ("see you", "later"),
        ("hello there", "general kenobi"),
        ("what's up", "not much"),
        ("welcome", "thank you"),
        ("ready", "go ahead"),
        ("hello", "hi"),
        ("greetings", "salutations"),
    ];
    // provider_b = "deepseek" — DeepSeek is OpenAI-wire-compatible, so the codec is
    // OpenAiCodec; routing through `deepseek` lets the REAL upstream fire when the key
    // is set, otherwise the mock falls back.
    pairs
        .iter()
        .enumerate()
        .map(|(i, (q, a))| SamplePair {
            label: format!("plain_text_{i:02}"),
            kind: "plain_text",
            provider_a: "anthropic",
            native_a: json!({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": [{"type": "text", "text": a}]}
                ]
            }),
            provider_b: "deepseek",
            native_b: json!({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            }),
        })
        .collect()
}

fn gen_tool_cycle() -> Vec<SamplePair> {
    let cases = [
        ("get_weather", "city", "SF", "sunny 22C"),
        ("get_weather", "city", "NYC", "cloudy 15C"),
        ("get_weather", "city", "London", "rain 12C"),
        ("get_stock", "symbol", "AAPL", "232.5"),
        ("get_stock", "symbol", "MSFT", "412.1"),
        ("translate", "text", "hi", "salut"),
        ("translate", "text", "bye", "ciao"),
        ("calc", "expr", "2+2", "4"),
        ("calc", "expr", "7*6", "42"),
        ("lookup", "key", "alpha", "first letter"),
        ("lookup", "key", "omega", "last letter"),
        ("ping", "host", "example.com", "ok"),
    ];
    cases
        .iter()
        .enumerate()
        .map(|(i, &(fn_name, arg, val, out))| {
            let mut arg_obj = serde_json::Map::new();
            arg_obj.insert(arg.to_string(), Value::String(val.to_string()));
            let args_json = Value::Object(arg_obj);
            let args_str = serde_json::to_string(&args_json).unwrap_or_else(|_| "{}".to_string());
            SamplePair {
                label: format!("tool_cycle_{i:02}"),
                kind: "tool_cycle",
                provider_a: "anthropic",
                native_a: json!({
                    "messages": [
                        {"role": "user", "content": format!("{fn_name}({val})?")},
                        {"role": "assistant", "content": [
                            {"type": "tool_use", "id": format!("toolu_{i:02}"), "name": fn_name, "input": args_json.clone()}
                        ]},
                        {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": format!("toolu_{i:02}"), "content": out}
                        ]}
                    ]
                }),
                provider_b: "responses",
                native_b: json!({
                    "input": [
                        {"role": "user", "content": [{"type": "input_text", "text": format!("{fn_name}({val})?")}]},
                        {"type": "function_call", "call_id": format!("fc_{i:02}"), "name": fn_name,
                         "arguments": args_str},
                        {"type": "function_call_output", "call_id": format!("fc_{i:02}"), "output": out}
                    ]
                }),
            }
        })
        .collect()
}

fn gen_multimodal() -> Vec<SamplePair> {
    // image base64 placeholder kept tiny — real samples would use real bytes.
    let mimes = ["image/png", "image/jpeg", "image/webp"];
    let prompts = [
        ("describe this", "a cat"),
        ("what is in the photo", "a dog"),
        ("identify this", "a chair"),
        ("read the sign", "stop"),
        ("count animals", "three"),
        ("describe scenery", "mountains"),
        ("describe building", "a tower"),
        ("read label", "fragile"),
        ("what color", "blue"),
        ("describe outfit", "red shirt"),
        ("identify plant", "a fern"),
        ("describe weather", "snowing"),
    ];
    prompts
        .iter()
        .enumerate()
        .map(|(i, (q, a))| SamplePair {
            label: format!("multimodal_{i:02}"),
            kind: "multimodal",
            provider_a: "anthropic",
            native_a: json!({
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": q},
                        {"type": "image", "source": {
                            "type": "base64", "media_type": mimes[i % mimes.len()], "data": "AAAA"
                        }}
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": a}]}
                ]
            }),
            provider_b: "gemini",
            native_b: json!({
                "contents": [
                    {"role": "user", "parts": [
                        {"text": q},
                        {"inlineData": {"mimeType": mimes[i % mimes.len()], "data": "AAAA"}}
                    ]},
                    {"role": "model", "parts": [{"text": a}]}
                ]
            }),
        })
        .collect()
}

fn gen_thinking() -> Vec<SamplePair> {
    // No signature → anthropic side and openai reasoning_content side share the same nf
    // (signature is the only field that diverges, and it is None on both sides here).
    let cases = [
        ("step by step", "answer is 7"),
        ("let me think", "the result is 12"),
        ("considering options", "option B"),
        ("first analyze", "conclusion: yes"),
        ("breaking it down", "three parts"),
        ("checking edge cases", "none found"),
        ("recall the rule", "use the formula"),
        ("eliminating impossible", "must be C"),
        ("counting carefully", "exactly 9"),
        ("rephrasing question", "what they want is X"),
        ("looking up context", "based on history"),
        ("reflecting on it", "the simpler path"),
    ];
    cases
        .iter()
        .enumerate()
        .map(|(i, (think, ans))| SamplePair {
            label: format!("thinking_{i:02}"),
            kind: "thinking",
            provider_a: "anthropic",
            native_a: json!({
                "messages": [
                    {"role": "user", "content": "solve this"},
                    {"role": "assistant", "content": [
                        {"type": "thinking", "thinking": think},
                        {"type": "text", "text": ans}
                    ]}
                ]
            }),
            provider_b: "deepseek",
            native_b: json!({
                "messages": [
                    {"role": "user", "content": "solve this"},
                    {"role": "assistant", "content": ans, "reasoning_content": think}
                ]
            }),
        })
        .collect()
}

fn all_samples() -> Vec<SamplePair> {
    let mut s = Vec::new();
    s.extend(gen_plain_text());
    s.extend(gen_tool_cycle());
    s.extend(gen_multimodal());
    s.extend(gen_thinking());
    s
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let out_dir = format!("{manifest_dir}/examples");
    fs::create_dir_all(format!("{out_dir}/anchor_witness")).ok();

    let samples = all_samples();
    println!("anchor_witness — {} sample pairs", samples.len());

    // Persist the sample set for reproducibility (no API keys, no responses, no PII).
    let samples_path = format!("{out_dir}/anchor_samples.json");
    fs::write(&samples_path, serde_json::to_string_pretty(&samples)?)?;
    println!("samples dumped to: {samples_path}");

    // Pick a single target provider as our observer. DeepSeek shares the OpenAI wire
    // shape, so we translate every sample to the openai codec before calling. Same
    // observer is the precondition for ε (we are NOT comparing different LLMs here; we
    // are comparing whether one LLM gives the same response to two same-nf inputs).
    let target_provider = "deepseek";
    let target_codec_name = "openai"; // deepseek wire is openai-compatible
    let target = upstream_for(target_provider).expect("deepseek preset known");

    let mut triples = Vec::new();
    let mut nf_match = 0u64;
    let mut nf_mismatch = 0u64;
    let mut upstream_skipped = 0u64;
    let mut kind_counts: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();

    for s in &samples {
        let m1 = lift(s.provider_a, &s.native_a)?;
        let m2 = lift(s.provider_b, &s.native_b)?;
        let entry = kind_counts.entry(s.kind).or_insert((0, 0));
        if m1 != m2 {
            nf_mismatch += 1;
            entry.1 += 1;
            continue;
        }
        nf_match += 1;
        entry.0 += 1;

        // Translate both natives to the target wire, then ask the same observer for a
        // response to each. The translate step ITSELF can introduce loss (recorded but
        // not used here); we measure the consumer-side gap on a same-nf pair.
        let mut target_a = match translate(s.provider_a, target_codec_name, &s.native_a) {
            Ok((v, _)) => v,
            Err(e) => {
                if upstream_skipped == 0 {
                    eprintln!("first translate(a) failure (kind={}): {e:?}", s.kind);
                }
                upstream_skipped += 1;
                continue;
            }
        };
        let mut target_b = match translate(s.provider_b, target_codec_name, &s.native_b) {
            Ok((v, _)) => v,
            Err(e) => {
                if upstream_skipped == 0 {
                    eprintln!("first translate(b) failure (kind={}): {e:?}", s.kind);
                }
                upstream_skipped += 1;
                continue;
            }
        };
        // Inject the model field — required by deepseek and by openai-compatible APIs
        // generally. Translate strips envelope fields by design; the routing layer
        // restores them.
        // Inject model + temperature 0 (greedy, deterministic) so same-prompt → same
        // response under a well-behaved LLM; any residual ε then reflects real codec /
        // translate loss, not LLM sampling noise.
        for v in [&mut target_a, &mut target_b] {
            if let Some(obj) = v.as_object_mut() {
                obj.entry("model".to_string()).or_insert_with(|| json!("deepseek-chat"));
                obj.insert("temperature".to_string(), json!(0));
                obj.insert("max_tokens".to_string(), json!(64));
            }
        }

        // Real HTTP failure on either side → skip the pair rather than abort. Mock is
        // total; this branch only matters for REAL mode.
        let resp_a = match target.call(target_a).await {
            Ok(v) => v,
            Err(e) => {
                if upstream_skipped == 0 {
                    eprintln!("first call(a) failure (kind={}): {e}", s.kind);
                }
                upstream_skipped += 1;
                continue;
            }
        };
        let resp_b = match target.call(target_b).await {
            Ok(v) => v,
            Err(_) => {
                upstream_skipped += 1;
                continue;
            }
        };
        let mut rv = ResponseVector::new();
        rv.insert(target_provider, resp_a, resp_b);
        triples.push((m1, m2, rv));
    }

    let eps = epsilon_estimate(&triples);

    let mode = format!(
        "anthropic={}, openai={}, gemini={}, deepseek={}",
        mode_label("anthropic"),
        mode_label("openai"),
        mode_label("gemini"),
        mode_label("deepseek"),
    );

    println!("mode: {mode}");
    println!("nf_match           = {nf_match}");
    println!("nf_mismatch        = {nf_mismatch}");
    println!("upstream_skipped   = {upstream_skipped}");
    println!("ε                  = {eps}");

    // Write RESULTS.md v2 per task-spec §10.3 sub-template A.
    let rustc = option_env!("RUSTC_VERSION").unwrap_or("rustc (compile-time unknown)");
    let now = std::env::var("CARGO_BUILD_TIMESTAMP").unwrap_or_else(|_| "build-time".to_string());

    let mut md = String::new();
    md.push_str("# Anchor Witness RESULTS v2\n\n");
    md.push_str(&format!("## Mode\n[MIXED@{}]\n\n", mode));
    md.push_str("## Environment\n");
    md.push_str(&format!("- rustc: {rustc}\n"));
    md.push_str("- agent-comm: 0.1.0\n");
    md.push_str("- c-meter: 0.1.0\n");
    md.push_str(&format!("- Generated: {now}\n\n"));
    md.push_str("## Sample Set\n");
    md.push_str(&format!("- Total pairs: {}\n", samples.len()));
    md.push_str("- Distribution:\n");
    for (k, (mat, mis)) in &kind_counts {
        md.push_str(&format!("  - {k}: {} (nf_match {}, nf_mismatch {})\n", mat + mis, mat, mis));
    }
    md.push('\n');
    md.push_str("## Codecs Tested\n");
    md.push_str(&format!("- anthropic ({})\n", mode_label("anthropic")));
    md.push_str(&format!("- openai ({})\n", mode_label("openai")));
    md.push_str(&format!("- gemini ({})\n", mode_label("gemini")));
    md.push_str(&format!("- deepseek ({})\n\n", mode_label("deepseek")));
    md.push_str("## Results\n");
    md.push_str(&format!("- ε (sup over observers): {eps}\n"));
    md.push_str(&format!("- nf_match / nf_mismatch / upstream_skipped: {nf_match}/{nf_mismatch}/{upstream_skipped}\n"));
    md.push_str(&format!("- target observer: `{target_provider}` (codec: `{target_codec_name}`)\n\n"));
    md.push_str("## Reproduce\n");
    md.push_str("```bash\ncargo run --example anchor_witness\n```\n\n");
    md.push_str("## Notes\n");
    md.push_str("- Mock providers hash request bytes; ε under a fully-mock fleet is 0 by\n");
    md.push_str("  construction (same nf → identical request normalization → identical hash).\n");
    md.push_str("- For any provider in REAL mode, ε > 0 is expected due to LLM stochasticity;\n");
    md.push_str("  the magnitude indicates how often same-nf inputs yield divergent outputs.\n");
    md.push_str("- Privacy: only `blake3(response_text)` is collected from real providers;\n");
    md.push_str("  raw response content is discarded inside the adapter.\n");

    let results_path = format!("{out_dir}/anchor_witness/RESULTS.md");
    fs::write(&results_path, md)?;
    println!("RESULTS written to: {results_path}");

    Ok(())
}
