# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-07-01

### Added
- Extension generators (Thinking, Media, Video) wire-frozen per SPEC v1.7 §2bis.
- SPEC v1.8 draft: cache_control vendor applicability table.

### Changed
- All `/// Wire EXPERIMENTAL` markers replaced with `/// Wire-frozen (SPEC v1.7 §2bis)`.
- README updated to reflect frozen extension generators.

### Fixed
- Workspace Cargo.toml: temporarily excluded `agent-dispatcher` whose
  `gml_arcadedb_compiler` path dependency is not available, restoring
  `cargo build` / `cargo test` across the monorepo.

## [0.1.0] — 2026-06-28

### Added
- Initial protocol Step1: neutral conversation IR (kernel K + normalize + validate).
- Step2 tooling: 4 reference codecs (Anthropic, OpenAI, Gemini, Responses),
  ⊥ (bottom) max-loss codec, conformance suite (121 tests).
- LossObligation accounting (R-3 never silent), dual envelope, fail-closed identity gates.
- Phi three-coordinate metrics (G/Σ/R) and empirical ε estimation.
- anchor_witness example for cross-vendor ε measurement.
