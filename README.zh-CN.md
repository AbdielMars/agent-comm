# agent-comm

**多厂商 LLM 对话的中立格式。在 Claude、GPT、Gemini、DeepSeek 之间无缝切换——没有格式锁定，不会静默丢失数据。**

一个 `Conversation` 类型，是**一个小核 K 之上内容生成器的余极限（colimit）**。每个厂商格式通过 `ProviderCodec` 对着这个 IR 做编解码——厂商之间**永不两两转换**。

---

## 解决什么问题

2024–2025 年，LLM 市场进入了碎片化的多厂商时代。Claude、GPT、Gemini、DeepSeek、Kimi、Qwen…每个都说自己的方言。如果你想让系统"对上不依赖单一厂商、对下能接各种模型"，你只有一个选择：

> **要么被一家厂商绑定；要么自己写 N² 条两两转换代码，而且永远不知道转换过程中丢了什么。**

传统的中转方案（CCswitch、one-api 等）做完格式转换后，不会告诉你——那个被你静默转掉的 `thinking` block 丢了、那个截断的 tool_call 参数被悄悄变成了 `{}`。

**agent-comm 的解法**：建一个中立 IR，所有厂商只对这个 IR 做编解码：

```
不是： Anthropic ──→ OpenAI          (n² 条腿，每加一个厂商成本翻倍)
       Gemini   ──→ DeepSeek
       ...

而是： Anthropic ──┐
       OpenAI    ──┼──→ 中立 IR ──→ 任何厂商
       Gemini    ──┘                   (2n 条腿，每加一个厂商只写一次)
```

---

## 快速开始

```bash
# 添加到你的 Cargo.toml
cargo add agent-comm

# 跨厂商翻译
use agent_comm::translate;
let openai_output = translate("anthropic", "openai", &claude_json)?;
```

零异步运行时、零网络依赖——纯数据变换。编解码通过 `encode` / `decode` 执行，两者都是 **fail-closed**（失败闭合）：未授权返回错误，绝不猜测。

---

## 核心能力

### v0 包含

| 能力 | 说明 |
|---|---|
| **核 K**（wire 已冻结） | `Text` / `ToolCall` / `ToolResult` |
| **扩展生成器**（wire 已冻结） | `Thinking` / `Media` / `Video` |
| **参考 Codec** | `AnthropicCodec` / `OpenAiCodec` / `GeminiCodec` / `ResponsesCodec` |
| **规范形 Normalize** | R5 → R1 → R6 → R3 → R2。幂等。两段不同厂商的对话语义相等当且仅当它们的规范形相等 |
| **损失记账 LossObligation** | 双向记账（up 和 down 都可能丢），每条损失类型化，绝不静默 |
| **跨厂商翻译** | `translate()` = `down_to ∘ up_from`，替代整族手写转换器 |
| **请求信封** | `split_envelope` / `apply_envelope` — 模型参数、用量、指纹不进入 IR 内核 |
| **Codec 注册表** | `codec_for()` — 别名归一化（`claude → anthropic` / `chat → openai` / `codex → responses`） |
| **往返一致性校验** | `check_round_trip()` — 验证 `up → normalize → down → up → normalize` 回到相同规范形 |
| **一致性测试套件** | `cargo test --features conformance` — 7 种 gate，PASS = 获 "agent-comm compatible" 印章 |
| **度量与 Φ 坐标** | 三维复杂度（Genesis / Stratification / Reification）+ 经验 ε 估计 |

---

## 设计理念

> 这些不是口号，是每个 codec 必须遵守的契约。一致性测试套件会验证它们。

### 绝不静默丢失（R-3）

如果你的 wire 载不动某个特性，**你必须说出来，不能悄悄地消化掉**。

一个截断的 JSON 参数字符串不允许变成空对象 `{}`——它必须是 `LossObligation { dropped_kind: "behav.truncated_args" }`。Reasonix 的 `closeTruncatedJSON → "{}"` 就是典型的反面教材——调用者永远不知道参数被吞了。

### 失败闭合（Fail-closed）

不确定的时候，拒绝而不是猜测。身份验证不过就不给编解码。输入格式不对就返回 `CodecError::Malformed`。没有"我帮你想一想可能是这个意思"。

### 双账本（Cor5.2）

行为丢失和计费丢失分开记：

| 账本 | 前缀 | 例子 |
|---|---|---|
| ≈_behav（行为） | `behav.*` | `behav.placement_collapsed`、`behav.truncated_args` |
| ≈_bill（计费） | `bill.*` | `bill.cache_directive_lost` |

缓存断点丢了（bill 损失）不等于你对话的内容出了错。两个账本各自独立审计。

### 厂商中立

在 `codec_for` 注册表里，Anthropic 和 Kimi、Claude 和 DeepSeek——它们在代码结构上完全对等。没有"Anthropic 的 codec 是核心，其他的额外加上去"这种事。厂商中立的数学含义是：对所有 provider 使用同一套判定函数。

---

## 扩展

### 接入你自己的厂商

复制模板 → 填 5 个 TODO → 注册 → 跑一致性套件 → 发 PR。

```bash
cp docs/codec_template.rs src/codecs/my_provider.rs
# 填 5 个 TODO
# 在 codec_for() 注册
cargo test --features conformance  # 通过即获印章
```

详见 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md) 和 [CODEC_BUILDER_GUIDE.md](docs/CODEC_BUILDER_GUIDE.md)。

### 添加一个生成器

添加一个 `Content` 变体 = 给 colimit 加一条腿。已有变体和它们的 wire 格式不动。

### 报告 Bug / 提 PR

看 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。

---

## 依赖

`serde`、`serde_json`、`blake3`。零外部协议依赖。无异步运行时，无网络。

---

## 证书

MIT OR Apache-2.0。
