# agent-comm 协议规格 v0.2.0

> **状态：稳定。** 内核 K 的 wire 格式（Text/ToolCall/ToolResult）及扩展生成器（Thinking/Media/Video）均已冻结。未来版本将保持向后兼容。
>
> **语言**：本文档以中文撰写，关键术语在首次出现时附英文原文。

---

## 1. 引言

### 1.1 背景

每个 LLM 厂商使用不同的 wire 格式：

| 厂商 | 工具调用格式 | 工具结果格式 | 推理内容格式 |
|---|---|---|---|
| Anthropic / Claude | `tool_use` block | 用户消息内的 `tool_result` block | `thinking` block |
| OpenAI / GPT | 助手消息上的 `tool_calls` 数组 | 独立的 `tool` 角色消息 | `reasoning_content` 字段 |
| Google / Gemini | `functionCall` part | `functionResponse` part | `thought` part |
| DeepSeek 等 | OpenAI 兼容的 `tool_calls` | OpenAI 兼容的 `tool` 角色 | `reasoning_content` 字段 |

这些差异意味着，一段在一个厂商格式内合理的对话，必须经过**转换**才能用于其他厂商。传统做法要么是：

- **两两转换**——O(n²) 的问题：每新增一个厂商，就要对所有已有厂商写正向和反向转换。转换过程中 `tool_call` 参数可能被静默丢弃或强制转换（例如截断的 JSON 字符串变成 `{}`）。
- **黑盒代理**——在中转过程中不告诉调用方丢了什么。没有审计记录。

### 1.2 协议解决什么问题

agent-comm 定义了一个**中立对话 IR**（中间表示）和一套 **codec** 接口。每个厂商对着这个 IR 做编解码——厂商之间永不两两转换：

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│ Anthropic │────▶│              │◀────│  OpenAI  │
└──────────┘     │   中立 IR    │     └──────────┘
┌──────────┐     │   + 规范形   │     ┌──────────┐
│  Gemini  │────▶│              │◀────│Responses │
└──────────┘     └──────────────┘     └──────────┘
```

核心保证：

- **每个丢失信息的转换都记录一条类型化的 LossObligation**——没有任何东西被静默丢弃。
- **两段对话语义相等当且仅当它们的规范形相等**——这是一个可判定的性质。
- **所有编解码操作都是失败闭合的**——未授权的访问被拒绝，不会悄悄绕过。

### 1.3 范围与边界

| 在范围内 | 不在范围内 |
|---|---|
| 对话结构（轮次、角色、内容块） | 网络传输（HTTP、WebSocket、SSE） |
| 工具调用与工具结果 | API 密钥管理 |
| 推理/思考内容 | 模型选择与路由 |
| 多模态媒体（图片、视频） | 运行时流式处理 |
| 跨厂商转换忠实度 | 用户身份认证（超出 Principal 的范畴） |
| 损失记账（行为 + 计费） | |

### 1.4 为什么用余极限？（数学基础）

> 本节解释协议的底层设计原理。你不必理解本节就能**使用** agent-comm，但如果你想理解它为什么这样工作，请继续阅读。

**问题**：每个厂商以不同的形状描述一段对话。Anthropic 有 `tool_use` 块，OpenAI 有 `tool_calls` 数组，Gemini 有 `functionCall` 部分。这些不是同一回事——但它们都试图表示同一个底层概念（"模型调用了一个函数"）。

如果做两两转换（Anthropic↔OpenAI、OpenAI↔Gemini 等等），你需要 n² 个转换器，而且没有一个单一的"事实来源"来定义一段对话究竟是什么。

**余极限的解法**：余极限是一种数学构造，它说："所有这些不同的形状只是同一个底层结构的不同视角。与其在视角之间直接翻译，不如每个视角都对底层结构做编解码。"

具体来说：

- 定义一个**小核 K**（kernel），包含最基本的对话元素：`Text`、`ToolCall`、`ToolResult`。
- 每个厂商的格式是 K 在该厂商原生 wire 格式上的一个**投影**（projection）。
- 中立 IR 就是**余极限**——把所有投影整合到一个统一表示中的结构。
- 新增一个厂商 = 向已有的余极限添加一个新的投影。已有的投影不需要改变。

**这给了我们什么**：

| 性质 | 余极限如何保证 |
|---|---|
| **无特权锚** | 所有厂商是对等的——每个都是同一个核 K 的一个投影 |
| **O(n) 而非 O(n²)** | 每个厂商一个 codec，而不是 n² 个两两转换器 |
| **语义相等** | 两个不同厂商的对话语义相等当且仅当它们的规范形（余极限的规范代表）相等 |
| **组合性** | `translate(A→C) = down_C ∘ up_A`——分解定理的代码形式 |

**延伸阅读**：形式化的处理（余极限定义、规范形合成证明、≈_behav / ≈_bill 等价类分离）在配套论文中。详情请参考项目的学术频道。

---

## 2. 术语

| 术语 | 定义 |
|---|---|
| **核 K**（Kernel K） | 已冻结的核心内容生成器集合：`Text`、`ToolCall`、`ToolResult`。每个 codec 必须支持这些。 |
| **扩展生成器**（Extension generator） | 可选但已冻结的内容类型：`Thinking`、`Media`、`Video`。不能表达的 codec 记录一条损失，而不是报错。 |
| **对话**（Conversation） | 一个有序的 `Turn` 序列，每个 `Turn` 包含一个 `Role` 和一个 `Content` 列表。 |
| **规范形**（Normal form）`nf(·)` | 一个确定性的规范化管线（R5→R1→R6→R3→R2）。规范形是唯一的：`nf(nf(x)) == nf(x)`。 |
| **损失义务**（LossObligation） | 一个类型化的记录，表示某个特性 codec 无法承载。双向：`up`（原生→IR）和 `down`（IR→原生）都可能产生损失。 |
| **信封**（Envelope） | 请求层或响应层的元数据（模型名、用量、指纹），位于 IR 内核之外。 |
| **失败闭合**（Fail-closed） | 调用者未授权时拒绝操作，而非悄悄降级。 |

---

## 3. 协议架构

协议在四个层级上运作，从 L1（最低）到 L4（最高）：

```
L4: 语义 IR       ─  Conversation、Content、Turn、Role
L3: 信封          ─  model、max_tokens、tools、usage、fingerprint
L2: 原生 wire     ─  厂商特定的 JSON（Claude messages[]、GPT messages[] 等）
L1: 传输          ─  HTTP、WebSocket、SSE（不在协议范围内）
```

**L4 是协议本体。** L3 是请求/响应元数据的载体。L2 是厂商偶然的格式，由 codec 转换到 L4。L1 明确不在范围内。

---

## 4. 核心数据类型

### 4.1 角色（Role）

每个 `Turn` 携带一个角色。协议定义了四个角色：

```
Role ::= System | User | Assistant | Tool
```

厂商有不同的命名方式。规范映射为：

| 规范角色 | 厂商别名 |
|---|---|
| `System` | `system`、`developer` |
| `User` | `user`、`human` |
| `Assistant` | `assistant`、`ai`、`model` |
| `Tool` | `tool`、`function` |

### 4.2 内容生成器（Content）

一个 `Content` 值是以下变体之一。前三个构成**核 K**（已冻结，所有 codec 必须支持）。后三个是**扩展生成器**（已冻结，codec 如果不能表达则记录一条损失）。

#### 4.2.1 核 K（已冻结，必须支持）

```
Content::Text { text: String, cache_control: Option<CacheControl> }
```

纯文本块。`cache_control` 携带提示缓存断点元数据，属于**计费等价类**（≈_bill，即它影响计费精度而非行为）。不能表达 `cache_control` 的 codec 必须发出 `billing.cache_directive_lost`。

```
Content::ToolCall { id: CallId, name: ToolName, args: Value }
```

工具/函数调用。`args` 是一个 JSON 对象，其键的顺序是厂商偶然的；规范形会确定性地排序键（R2）。截断或无法解析的 `args` 字符串**不能**静默地变成 `{}`——codec 必须发出 `behavior.truncated_args`。

```
Content::ToolResult { ref_id: CallId, payload: Vec<Content>, cache_control: Option<CacheControl> }
```

工具结果。`ref_id` 必须指向一个先前的 `ToolCall.id`（由 `Conversation::validate` 校验）。N1 约束：`payload` 不能包含 `ToolCall`、`Thinking` 或 `Video`；嵌套深度 ≤ 2。

#### 4.2.2 扩展生成器（已冻结，可能产生损失）

```
Content::Thinking { text: String, sig: Option<String>, placement: Placement }
```

推理/思维链。`placement` 记录了推理内容在原生 wire 中的位置（它属于**行为**等价类——改变 placement 会改变运行时行为，例如 Letta 的工具分发）。从 `ToolCallKwargs` 折叠 placement 会产生 `behavior.placement_collapsed`。

```
Content::Media { mime: String, data: String }
```

内联 base64 媒体（image/* 或 application/pdf）。允许出现在 `ToolResult.payload` 中。

```
Content::Video { source: VideoSource, mime: String, duration_seconds: Option<u64> }
```

视频内容，支持 URL 或 base64 两种来源。N1：**不允许**出现在 `ToolResult.payload` 中。

### 4.3 位置（Placement）

推理位置（G1 裁定）：

| 位置 | 原生位置 | 示例厂商 |
|---|---|---|
| `Block` | 独立的推理块 | Anthropic `thinking`、Gemini `thought` |
| `ToolCallKwargs` | 嵌入在工具调用参数中 | Letta `put_inner_thoughts_in_kwargs` |
| `InlineString` | 行内字符串字段 | DeepSeek `reasoning_content` |

### 4.4 规范形（Normal Form）

对话 `c` 的规范形 `nf(c)` 通过按**顺序**应用以下管线计算：

| 步骤 | 名称 | 效果 |
|---|---|---|
| **R5** | 丢弃空文本 | 移除空的 `Text` 块；递归处理 `ToolResult.payload` |
| **R1** | 合并相邻文本 | 如果相邻 `Text` 块的 `cache_control` 相同，则合并 |
| **R6** | 工具角色规范 | 将每个 `ToolResult` 提升到独立的 `Role::Tool` 轮次，保持顺序 |
| **R3** | ID 规范 | 按首次出现顺序重命名 `ToolCall.id` 为 `call_0, call_1, ...` |
| **R2** | 参数键排序 | 递归排序 `ToolCall.args` 对象的键（通过 BTreeMap 重建） |

性质：

- **幂等**：`nf(nf(c)) == nf(c)`
- **语义相等**：`c1.semantic_eq(c2)` 当且仅当 `nf(c1) == nf(c2)`
- **确定性**：相同输入始终产生相同输出

---

## 5. 损失记账（Loss Accounting）

### 5.1 损失义务（LossObligation）

当 codec 无法将一个特性从一种表示传递到另一种时，它记录一条类型化的义务。损失是**双向的**：`up`（原生→IR）和 `down`（IR→原生）都可能产生损失。

```
LossObligation {
    provider: String,         // 例如 "openai"、"gemini"
    dropped_kind: String,    // 类型化的损失族
    turn_index: usize,       // 损失发生的位置
    recoverable: bool,       // 该特性在返回途中能否恢复？
    note: String,            // 可读的描述
    detail: Option<BTreeMap<String, String>>,  // 结构化的类型化负载
}
```

### 5.2 双账本分离

损失分为两个族，独立跟踪：

| 前缀 | 账本 | 示例 |
|---|---|---|
| `behavior.*` | **行为账本**（≈ 用户体验） | `behavior.truncated_args`、`behavior.placement_collapsed` |
| `billing.*` | **计费账本**（≈ 成本/用量） | `billing.cache_directive_lost` |

`behavior.*` 损失意味着消费者可能观察到不同的行为。`billing.*` 损失意味着计费精度可能受影响——但行为本身不变。

### 5.3 R-3 规则（绝不静默丢弃）

> **如果你的 wire 无法承载某个内核特性，你必须记录一条 LossObligation。绝对不能静默丢弃、强制转换到默认值、或伪造一个占位符。**

---

## 6. 信封（Envelope）

### 6.1 请求信封

不进入 IR 内核的请求层字段：

| 字段 | 说明 |
|---|---|
| `model` | 模型标识符（例如 `claude-3-5-sonnet`） |
| `max_tokens` | 最大输出标记数 |
| `temperature` / `top_p` | 采样参数 |
| `tools` | 工具/函数定义 |

`split_envelope` 函数将原生请求拆分为（对话 IR，信封）。`apply_envelope` 函数在 `down` 之后重新附加信封字段。

### 6.2 请求指纹（G8）

原生请求体的规范哈希：

```
request_fingerprint = blake3(canonical_sort(native_body))
```

对象键在哈希前被确定性地排序。相同的请求体在不同 JSON 序列化下 → 相同的指纹。

### 6.3 响应信封

响应侧的元数据：

| 字段 | 说明 |
|---|---|
| `echoed_model` | 厂商自报的模型 ID |
| `usage` | 用量统计（input_tokens、output_tokens 等） |
| `stop_reason` | 终止原因 |
| `response_fingerprint` | 响应体的规范哈希 |

### 6.4 缓存新鲜度（G9）

| 值 | 含义 |
|---|---|
| `Fresh` | 厂商确认缓存响应仍匹配当前真实状态 |
| `Stale` | 厂商标记缓存已过期但已服务响应 |
| `Unknown` | 无新鲜度信号（保守默认值） |

---

## 7. 一致性（Conformance）

### 7.1 Codec 契约

每个 `ProviderCodec` 实现必须满足：

```
check_round_trip(codec, native)  ⟹  nf(up(codec, native)) == nf(up(codec, down(codec, nf(up(codec, native)))))
```

用语言描述：将一段原生消息通过 IR 再转回，然后规范化，必须得到和直接转一次再规范化相同的规范形。

### 7.2 一致性门

一个符合规范的实现通过套件中的所有门：

| 门 | 检查什么 |
|---|---|
| **Round-trip** | 核 K 上的无损往返 |
| **Orphan tool_result (#20)** | 没有对应调用的工具结果被如实上报，而不是被丢弃 |
| **Abandoned tool_call (#19)** | 中断的调用被如实上报，而不是被伪造结果 |
| **Truncated args (#21)** | 截断的 JSON 参数发出 `behavior.truncated_args`，而不是静默 `{}` |
| **Model identity (#16)** | `echoed_model ≠ requested_model` → 路由变更发现 |
| **Face purity (#17)** | 用量字段属于声明的厂商面 |

### 7.3 底基线 Codec（⊥）

底基线 codec 是**最大损失基线**：它只表达 `Text`，其他所有内容都以类型化损失丢弃。它是一个参考底层：任何 codec 如果比 ⊥ 损失更多，说明有 bug。

---

## 8. 安全

### 8.1 失败闭合操作

每个 `encode` 和 `decode` 调用要求一个经过授权的 `Principal`：

```
encode(conv, codec, who: &Principal, gate: &dyn IdentityGate) -> Result<...>
decode(native, codec, who: &Principal, gate: &dyn IdentityGate) -> Result<...>
```

如果门不授权该主体，操作返回 `CommError::Unauthorized`。不存在绕过路径。

### 8.2 数据处理

- 不存储原始内容——只存储 `response_fingerprint`（哈希）和聚合的用量计数。
- `anchor_witness` 示例只从真实厂商收集 `blake3(response_text)`；原始响应文本不会离开适配器范围。

---

## 9. 版本

| 版本 | 状态 | 变更 |
|---|---|---|
| 0.1.0 | 已被取代 | 初始协议 Step1 + Step2 工具 |
| 0.2.0 | **当前稳定** | 核 K 冻结、扩展生成器冻结、信封双端、LossObligation v0.3、一致性套件定型 |

未来版本（≥ 0.3.0）将对核 K wire 格式保持向后兼容。扩展生成器可能新增变体，但已有变体不会被移除或重定义。

---

## 10. 参考

- `src/lib.rs` — IR、normalize、encode/decode、translate 的参考实现
- `src/protocol.rs` — Principal 和 IdentityGate 契约
- `src/codecs/` — 参考 codec（Anthropic、OpenAI、Gemini、Responses、Bottom）
- `src/conformance.rs` — 一致性套件运行器和参考向量
- `tests/conformance.rs` — 100+ 项一致性测试
- `CONTRIBUTING.zh-CN.md` — 如何添加新的 codec
- `CODEC_BUILDER_GUIDE.md` — Codec 作者的分步指南

---

> 规格版本 0.2.0。证书：MIT OR Apache-2.0。
