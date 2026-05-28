# Changelog

## [0.1.10](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.9...agentforge-v0.1.10) (2026-05-27)


### Features

* add AWS Bedrock provider with Converse API and SigV4 request signing ([c435a4e](https://github.com/bhavinkotak/agentforge/commit/c435a4e))
* fix CLI provider routing for ollama and bedrock arms ([c435a4e](https://github.com/bhavinkotak/agentforge/commit/c435a4e))
* add 15 integration tests for shadow-runs, finetune-export, benchmarks, and agent-runs routes ([c435a4e](https://github.com/bhavinkotak/agentforge/commit/c435a4e))
* update README with Bedrock provider docs, new API endpoints, Multi-Agent Testing, and Benchmark Suites sections ([c435a4e](https://github.com/bhavinkotak/agentforge/commit/c435a4e))

## [0.1.9](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.8...agentforge-v0.1.9) (2026-05-24)


### Bug Fixes

* **ci:** fix Docker .sqlx path and aarch64 musl cross-compilation ([14d77f6](https://github.com/bhavinkotak/agentforge/commit/14d77f6442051aa462b8e2dcc5557dab71addf54))

## [0.1.8](https://github.com/bhavinkotak/agentforge/compare/agentforge-v0.1.7...agentforge-v0.1.8) (2026-05-24)


### Features

* add auto-optimize toggle to eval run form ([7f595ea](https://github.com/bhavinkotak/agentforge/commit/7f595ea69d001dc28675650e68264e52685c4b6a))
* add OllamaClient, configure Ollama as default provider, fix duplicate SHA in opt loop ([f647edb](https://github.com/bhavinkotak/agentforge/commit/f647edb2288ae48f9f3e2fc767fff3d47c752c4a))
* API key auth, PATCH/scenarios endpoints, README + action.yml overhaul ([87c4aab](https://github.com/bhavinkotak/agentforge/commit/87c4aabefe3756fbf83e80bd5569ab72f13d0166))
* diff tab dropdowns, run→agent links, parent_sha, fmt fix, 100 unit tests ([c684409](https://github.com/bhavinkotak/agentforge/commit/c684409316932089dc420fa3b2e7efcbeaca0f0d))
* **diff:** implement real line-by-line unified diff for system prompt comparison ([7632ce6](https://github.com/bhavinkotak/agentforge/commit/7632ce6aac0eeaa3bc489ba938400db35bae570a))
* **diff:** show 'no content differences' message when versions are identical ([b9c8cfc](https://github.com/bhavinkotak/agentforge/commit/b9c8cfc74c04b8a2a87e5a62d5be630dc580a389))
* generate realistic, domain-aware test scenarios instead of vague placeholders ([69edce4](https://github.com/bhavinkotak/agentforge/commit/69edce4f0c1954cacdff3d2ad79101da64f6a3ce))
* iterative self-improvement loop with 95% threshold ([36c13b3](https://github.com/bhavinkotak/agentforge/commit/36c13b34e238e97f3aa58eac9ee610493593fe67))
* self-improving optimizer loop + 200+ regression tests ([b717fb3](https://github.com/bhavinkotak/agentforge/commit/b717fb32fd73bfd50ebde763a321d1efb5d0a10f))
* SSE progress endpoint, --watch mode, README stale version fixes ([9f8b814](https://github.com/bhavinkotak/agentforge/commit/9f8b814e25538eb90298adcd791da19c686bc9ea))
* **ui:** group agents by name on list page; add version history panel ([a57ab73](https://github.com/bhavinkotak/agentforge/commit/a57ab73778ee01ccd7e2992604240a8f07325b0d))


### Bug Fixes

* add api_error to DB enum, 60s timeout for scenario gen, improve error banner ([ca2910e](https://github.com/bhavinkotak/agentforge/commit/ca2910ee0accf1f76bc2a05d9e8e26ae50a744a1))
* api_error cluster for errored traces, clippy fixes, README/gitignore updates ([f7832ce](https://github.com/bhavinkotak/agentforge/commit/f7832cefd3f990b5b60160950db28b4c8a7c0bed))
* **ci:** refactor too_many_args to structs, fix deny.toml v2, update README ([f69a6dd](https://github.com/bhavinkotak/agentforge/commit/f69a6dd95e0bd2c0a61ce30eb4556a805181dd4b))
* cluster ordering bug, missing aggregate_score in list API, schema regression tests ([5a9da2e](https://github.com/bhavinkotak/agentforge/commit/5a9da2e919a70aa62a8f1ccc90268f3ee27473e3))
* correct NVIDIA error provider labels, surface trace failure_reason in UI ([cbe5df5](https://github.com/bhavinkotak/agentforge/commit/cbe5df5c1bf8055d950e6941e625e6ad60be03ba))
* **diff:** correct LCS edit-list traversal direction ([6ffe055](https://github.com/bhavinkotak/agentforge/commit/6ffe0558299b382f2e7cce8d3905a430d48de700))
* disable parallel_tool_calls for NVIDIA provider (single tool call only) ([5feca15](https://github.com/bhavinkotak/agentforge/commit/5feca1597fe90f7707e0b2ffdb65dfa273dd6e36))
* make optimizer produce real LLM-rewritten agent variants ([9bbdc1e](https://github.com/bhavinkotak/agentforge/commit/9bbdc1e604e30e294eb03e3e1491202742ef185b))
* **optimizer:** bump version on reorder_instructions, expose bump_patch_version_pub, add changelog to AgentResponse ([4f6e823](https://github.com/bhavinkotak/agentforge/commit/4f6e823c01d6d2470930db03ae19ca616e6de389))
* **optimizer:** rewrite tool description mutation to use compact description-only format ([33b37be](https://github.com/bhavinkotak/agentforge/commit/33b37be27e3543e71de408b5f8c6d3e6ee4432a9))
* pass actual run_id to RunnerConfig and derive scorer credentials from provider ([d946cbb](https://github.com/bhavinkotak/agentforge/commit/d946cbbcd1049627944765c9a2d8ac6c91abe80a))
* proper tool schemas + llama-3.1-70b for reliable function calling ([7f6b343](https://github.com/bhavinkotak/agentforge/commit/7f6b34395c0d442a53644aa7fdbdb31478aa169b))
* realistic mock tool responses for GitHub Actions tools ([e6d1ab7](https://github.com/bhavinkotak/agentforge/commit/e6d1ab7226784323b4b2e75eda029bf71c29893b))
* remove invalid ignoreDeprecations 6.0 from tsconfig (TS 5.9 compat) ([aa535c3](https://github.com/bhavinkotak/agentforge/commit/aa535c399eb2da6546eab70862fcb716508f149e))
* rustfmt formatting + move sqlx cache to crate-level .sqlx/ ([95fd249](https://github.com/bhavinkotak/agentforge/commit/95fd249740767904e1fae901a431573d7d585914))
* smarter failure cluster classification using weakest-dimension fallback ([34b3723](https://github.com/bhavinkotak/agentforge/commit/34b3723d6056f676a7ba72369936b61a03343fb6))
* suppress spurious postgres errors in CI test runs ([51d1f27](https://github.com/bhavinkotak/agentforge/commit/51d1f274652222d9cf3ed8bcf8676299506936b8))
* **tests:** update InstructionReorder test assertions to match 'Key Behavioral Rules' prefix ([e036ddf](https://github.com/bhavinkotak/agentforge/commit/e036ddf1fa2907f6e509f0b42148915732ca7250))
* vite proxy must rewrite /api/* to /api/v1/* not strip the prefix ([4f92a05](https://github.com/bhavinkotak/agentforge/commit/4f92a0519529da8b002c18e9ee49de2190d4887e))


### Documentation

* add Ollama and NVIDIA NIM setup docs, fix config table ([295697a](https://github.com/bhavinkotak/agentforge/commit/295697afcaf78e8043eae78100fc664562277762))

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
