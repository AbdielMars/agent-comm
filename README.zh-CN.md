# agent-comm

一个中立对话 IR：一个 `Conversation` 类型，是**一个小核（kernel）之上内容生成器的余极限（colimit）**。每个厂商格式通过 `ProviderCodec` 对着这个 IR 做编解码——厂商之间永不两两转换。

## v0 有什么

- **核 K**：`Text` / `ToolCall` / `ToolResult`（wire 已冻结）。
- **扩展生成器**（wire 已冻结，SPEC v1.7 §2bis）：`Thinking` / `Media` / `Video`。
- **Codec 实现**：`AnthropicCodec` / `OpenAiCodec` / `GeminiCodec` / `ResponsesCodec`。
- **规范形** `Conversation::normalize` — 独特的 nf(·)，计算管线为 **R5 → R1 → R6 → R3 → R2**：
  - R5 丢弃空文本 · R1 合并相邻文本 · R6 将 `tool_result` 折叠进独立的 Tool 轮
    （消除厂商偶发的角色放置差异）· R3 规范 id 为 `call_0, call_1, …` ·
    R2 排序 `ToolCall.args` 的 object key。
  - 幂等：`nf(nf(x)) == nf(x)`。两段对话语义相等当且仅当它们的规范形相等
    （`Conversation::semantic_eq`）。
- **校验** `Conversation::validate` — N1（`tool_result.payload` 不可包含
  `ToolCall` / `Thinking` / `Video`；嵌套深度 ≤ 2）、tool 链接闭合（每个
  `tool_result.ref` 必须指向前面的 `tool_call.id`）、`ToolCall` id 唯一性。
- **损失记账**——双向 `LossObligation`：`up`（厂商原生→IR）和 `down`
  （IR→厂商原生）都可能报告损失；不能承载的特性记录为类型化的义务，
  **绝不静默丢弃**。
- **跨厂商翻译** `translate(from, to, native)` =
  `down_to ∘ up_from`（分解定理的代码形式）。替代整族手写的两两 `X_to_Y` 转换器。
- **请求信封** `split_envelope` / `apply_envelope` — 请求层字段
  （`model` / `max_tokens` / `tools` / …）永不进入 IR 内核；信封层保留它们。
- **Codec 注册表** `codec_for(provider)` — 别名归一化
  （`claude → anthropic` / `chat → openai` / `codex → responses` / …）。
- **往返一致性校验** `check_round_trip(codec, native)` — 验证
  `up → normalize → down → up → normalize` 产生的规范形相同。
- **身份** `encode` / `decode` 是 **fail-closed**（失败闭合）：每个调用都需要一个经过
  `agent_protocol::IdentityGate` 授权的 `Principal`；未授权 → `CommError::Unauthorized`。

## 扩展

添加一个生成器 = 添加一个 `Content` 变体 = 给 colimit 加一条腿；已有变体和它们的 wire 格式不变。添加一个厂商 = 实现一个 `ProviderCodec` + 在 `codec_for` 注册一条分支；其他 codec 不受影响（没有特权锚——所有厂商是对等的）。

详见 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md) 和 [docs/CODEC_BUILDER_GUIDE.md](docs/CODEC_BUILDER_GUIDE.md)。

## 依赖

`agent-protocol`（标准契约）、`serde`、`serde_json`、`blake3`。无异步运行时、无网络依赖。

## 证书

MIT OR Apache-2.0。
