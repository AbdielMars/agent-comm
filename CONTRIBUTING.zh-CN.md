# 为 agent-suite 做贡献

感谢你有兴趣贡献代码。以下是如何开始。

## 快速开始

1. **Fork** 这个仓库。
2. **Clone** 你的 fork。
3. `cargo build --workspace` — 应该全部编译通过。
4. `cargo test --workspace` — 所有测试应该通过。
5. 做你的改动。
6. `cargo test --workspace` — 确认没有破坏任何东西。
7. **Push** 到你的 fork，然后开一个 **Pull Request**。

## 添加一个新的厂商 codec

这是最常见的贡献类型。添加一个厂商（比如 Kimi、Cohere 或其他）= 给 colimit 加一条腿。

1. 复制 `docs/codec_template.rs` 到 `src/codecs/<你的厂商>.rs`。
2. 填写 5 个 TODO 标记——把你的厂商 wire 格式映射到 kernel K。
3. 在 `src/codecs/mod.rs` 注册（`mod <你的厂商>; pub use <你的厂商>::<你的厂商>Codec;`）。
4. 在 `src/lib.rs` 的 `codec_for` 函数里注册。
5. 用你的厂商 wire 格式写参考向量（参考 `src/conformance.rs` 中的例子）。
6. 跑 `cargo test --features conformance`。通过 = 拿到 "agent-comm compatible" 印章。
7. 开 PR。在 PR 描述中附上示例请求/响应 payload，方便维护者验证你的映射。

**所有 codec 必须遵守的规则：**

- **绝不静默丢弃（R-3）。** 如果你的 wire 无法承载某一 kernel 生成器，必须发射一条类型化的 `LossObligation`。不允许静默返回 `{}`，不允许伪造占位符。
- **失败闭合（Fail-closed）。** 畸形输入 → `Err(CodecError::Malformed(..))`。不要猜测，不要掩盖。
- **测试一致性。** 套件中的每个门（gate）都必须 PASS。

## 添加一个新的一致性测试门

如果你发现有一类错误现有套件没有覆盖：

1. 在 `src/check.rs` 或 `src/conformance.rs` 中添加门。
2. 添加有区分度的测试向量——可信 codec 能通过、有损 codec 会失败。
3. 跑完整套件。

## 报告 Bug

开一个 issue，包含：
- 你期望发生什么
- 实际发生了什么
- 可以复现问题的最小原生 JSON payload
- `cargo test --features conformance` 的输出（如果相关）

## Pull Request checklist

- [ ] `cargo test --workspace` 通过
- [ ] `cargo test --features conformance` 通过
- [ ] 新代码有测试覆盖
- [ ] 公开 API 变更已写文档
- [ ] Codec 变更包含参考向量

## 证书

贡献即表示你同意你的贡献将使用与项目相同的条款（MIT OR Apache-2.0）授权。
