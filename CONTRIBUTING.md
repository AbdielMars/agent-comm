# Contributing to agent-suite

Thanks for considering contributing. Here's how to get started.

## Quick start

1. **Fork** this repository.
2. **Clone** your fork.
3. `cargo build --workspace` — everything should compile.
4. `cargo test --workspace` — all tests should pass.
5. Make your change.
6. `cargo test --workspace` again — make sure nothing broke.
7. **Push** to your fork and open a **Pull Request**.

## Adding a new provider codec

This is the most common contribution. Adding a provider (e.g. Kimi, Cohere, DeepSeek native) = adding one leg of the colimit.

1. Copy `docs/codec_template.rs` into `src/codecs/<your_provider>.rs`.
2. Fill the five `TODO` markers — map your provider's wire shape onto kernel K.
3. Register in `src/codecs/mod.rs` (`mod <your_provider>; pub use <your_provider>::<YourProvider>Codec;`).
4. Register in `codec_for` in `src/lib.rs`.
5. Write reference vectors in your provider's wire shape (see `src/conformance.rs` for examples).
6. Run `cargo test --features conformance`. If it passes, you earned the "agent-comm compatible" stamp.
7. Open a PR. Include sample request/response payloads in the PR description so reviewers can verify your mapping.

**Rules all codecs must follow:**

- **Never drop silently (R-3).** If your wire cannot express a kernel generator, emit a typed `LossObligation`. No silent `{}`, no fabricated placeholders.
- **Fail closed.** Malformed input → `Err(CodecError::Malformed(..))`. Don't guess, don't paper over.
- **Test conformance.** Every gate in the suite must PASS.

## Adding a conformance gate

If you find a class of errors that the existing suite doesn't catch:

1. Add the gate to `src/check.rs` or `src/conformance.rs`.
2. Add discriminating test vectors that a *faithful* codec passes but a *dropping* codec fails.
3. Run the full suite.

## Reporting bugs

Open an issue with:
- What you expected to happen
- What actually happened
- A minimal native JSON payload that reproduces the problem
- `cargo test --features conformance` output (if relevant)

## Pull Request checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo test --features conformance` passes
- [ ] New code has tests
- [ ] Public API changes are documented
- [ ] Codec changes include reference vectors

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).
