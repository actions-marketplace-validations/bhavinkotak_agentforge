# Changelog

## [0.1.7](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.6...agentforge-v0.1.7) (2026-05-20)


### Bug Fixes

* Docker src/ copy, aarch64 cross-compiler + Dependabot dep bumps ([a33f935](https://github.com/bhavinkotak/agentforge/commit/a33f935894b4275d961c0d01fb93ea05ef6328d5))

## [0.1.6](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.5...agentforge-v0.1.6) (2026-05-20)


### Features

* add NVIDIA NIM provider + permissions blocks in all workflows ([13b47e4](https://github.com/bhavinkotak/agentforge/commit/13b47e41bfde73c7322f424d6933c53e8b734c57))


### Bug Fixes

* address 4 remaining audit issues ([#11](https://github.com/bhavinkotak/agentforge/issues/11) DELETE endpoints, [#13](https://github.com/bhavinkotak/agentforge/issues/13) --max-cost, [#16](https://github.com/bhavinkotak/agentforge/issues/16) rate limiting, tests) ([497e65a](https://github.com/bhavinkotak/agentforge/commit/497e65acd04de42cc94ee57b28c7f49023126580))
* address all 22 audit issues ([2b12820](https://github.com/bhavinkotak/agentforge/commit/2b12820550697e272e679376f36dd0b6eef6448a))
* CI clippy + address Dependabot Cargo PRs ([87b85f9](https://github.com/bhavinkotak/agentforge/commit/87b85f9f0475868b84ccc094cee679f6555dd0a4))
* **ci:** capture binary exit code with || EVAL_EXIT=$? to bypass bash errexit ([689142e](https://github.com/bhavinkotak/agentforge/commit/689142ee3b96531a8e61dd41dbbb728c82411cf0))
* **ci:** fix agent-test-nvidia.yml — secrets context not available in job if condition ([cd5be8f](https://github.com/bhavinkotak/agentforge/commit/cd5be8f1d6cf230da7250605d5458743a6ff824d))
* **nvidia:** switch to 70b model + accept exit-code 1 as connectivity-confirmed in CI ([c0456b1](https://github.com/bhavinkotak/agentforge/commit/c0456b115655d65427f6e7dfb680f2c5dfdf8125))
* **nvidia:** switch to mistralai/mistral-small-4-119b-2603 (llama-3.1-70b removed from NIM) ([55492cc](https://github.com/bhavinkotak/agentforge/commit/55492cc005271397fb05cc3b78a4bb04249549e7))
* pass --provider/--judge-provider nvidia flags in agent-test workflow ([c521546](https://github.com/bhavinkotak/agentforge/commit/c521546326f5ea38bc6a90e704e05f96ea8f04de))
* **runner:** fix multi-turn tool calls for vLLM backends (NVIDIA NIM Mistral) ([5b3fbf2](https://github.com/bhavinkotak/agentforge/commit/5b3fbf21e7bcb79a596189e1dea3eea547ff03b4))
* **runner:** NvidiaClient overrides request model with configured NVIDIA model ([d8a9500](https://github.com/bhavinkotak/agentforge/commit/d8a9500c930c58144217f05701d5b912796e2645))
* **runner:** rename ToolCall tool_type -&gt; type in serde (fix multi-turn tool call failure); add AGENTFORGE_DEBUG workflow mode ([4c22c7e](https://github.com/bhavinkotak/agentforge/commit/4c22c7e946319642457d79cbc3b0a3c714ac9020))
* **runner:** robust tool-call parsing and empty-list guard for vLLM backends ([9d2f29f](https://github.com/bhavinkotak/agentforge/commit/9d2f29f485e925ed4122c94c438d9d4a3314a4a0))
* third round of audit issues — API list endpoints, CLI --agent-format, pagination, doc gaps ([938fe74](https://github.com/bhavinkotak/agentforge/commit/938fe748bde57f78fbd17d4260c0c192244e8e85))

## [0.1.5](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.4...agentforge-v0.1.5) (2026-05-01)


### Bug Fixes

* build issues ([e3c3466](https://github.com/bhavinkotak/agentforge/commit/e3c34662fef0f80e599d18fc13e2a23c1847518f))
* update Cargo.lock to fix --locked build failures ([d6590c8](https://github.com/bhavinkotak/agentforge/commit/d6590c8074dfd35b313aad4ce969bcc21933fbcb))

## [0.1.4](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.3...agentforge-v0.1.4) (2026-04-30)


### Features

* add React UI, new API routes, migrations, and test script ([0eb4ac8](https://github.com/bhavinkotak/agentforge/commit/0eb4ac86c5e3a8124190c79caa88b8c6e5df6b12))
* **ui:** register agent UX improvements and scorecard error banner ([bdb2c69](https://github.com/bhavinkotak/agentforge/commit/bdb2c69107949d6187079acee004b10ea2d27f46))


### Bug Fixes

* completed_count/error_count never written + GitHub URL auto-resolve ([b2446e6](https://github.com/bhavinkotak/agentforge/commit/b2446e6fd37da6ffb4485f2b69b10250f703e10c))

## [0.1.3](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.2...agentforge-v0.1.3) (2026-04-30)


### Bug Fixes

* shorten action.yml description to meet GitHub Marketplace 125-char limit ([e0b5661](https://github.com/bhavinkotak/agentforge/commit/e0b566111f242dfcd25d143e53baf5623dcf1aad))

## [0.1.2](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.1...agentforge-v0.1.2) (2026-04-30)


### Documentation

* add CODEOWNERS, CONTRIBUTING guide, and GitHub Actions marketplace section ([05077d1](https://github.com/bhavinkotak/agentforge/commit/05077d1cef666bf53b58bdbf84f64fbd9b72a9f3))

## [0.1.1](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.0...agentforge-v0.1.1) (2026-04-30)


### Features

* add GitHub Copilot .agent.md format support ([0a595bf](https://github.com/bhavinkotak/agentforge/commit/0a595bf01a96b236262758c14b6218ca9c3f5354))
* initial implementation of AgentForge ([f2af419](https://github.com/bhavinkotak/agentforge/commit/f2af419920628e69fb6ec9dc5c45b020310d1d62))


### Bug Fixes

* add explicit toolchain input to dtolnay/rust-toolchain SHA-pinned calls ([5782635](https://github.com/bhavinkotak/agentforge/commit/5782635371df6a3c57a530dd0bcaf2804a79c24b))
* enable release-please to version workspace correctly ([a73e45b](https://github.com/bhavinkotak/agentforge/commit/a73e45b1efef4e336caf35b5750072da70ecd65c))
* pin all Actions to SHA, add missing agentforge-api crate, fix cargo audit ([d4960c2](https://github.com/bhavinkotak/agentforge/commit/d4960c23dc172af3d7b0d54622e54cee43d45858))
* pin all Actions to SHA, add missing crate, fix cargo audit ([8afe4a2](https://github.com/bhavinkotak/agentforge/commit/8afe4a23d5132f299252bee952b0faee315d1830))
* resolve all clippy -D warnings and rustfmt issues ([2db1fe2](https://github.com/bhavinkotak/agentforge/commit/2db1fe24bbefae0014926185401554c65230bed3))
* use explicit version = "0.1.0" in all workspace member crates ([438f267](https://github.com/bhavinkotak/agentforge/commit/438f2674d83c01f8f9bd150ce9b24ec741c798f9))
