# AgentForge

> **One file in. A better agent out.**

[![CI](https://github.com/bhavinkotak/agentforge/actions/workflows/ci.yml/badge.svg)](https://github.com/bhavinkotak/agentforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/bhavinkotak/agentforge?sort=semver)](https://github.com/bhavinkotak/agentforge/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Marketplace](https://img.shields.io/badge/GitHub%20Actions-Marketplace-blue?logo=github)](https://github.com/marketplace/actions/agentforge-eval)

AgentForge is a self-improving AI agent optimization platform written in Rust. Feed it a single agent file — a declarative spec describing your AI agent's system prompt, tools, output schemas, and behavioral constraints — and it autonomously generates test scenarios, runs the agent, scores every execution trace, and iterates on the specification until it converges on a measurably better version.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Agent File Format](#agent-file-format)
- [CLI Usage](#cli-usage)
- [REST API](#rest-api)
- [Scoring Dimensions](#scoring-dimensions)
- [Promotion Gatekeeper](#promotion-gatekeeper)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [GitHub Actions Marketplace](#github-actions-marketplace)
- [CI/CD Integration](#cicd-integration)
- [Contributing](#contributing)
- [Roadmap](#roadmap)

---

## Overview

AI agent development has a painful quality gap: teams ship prompts and tool definitions with little systematic testing, and improvements are made based on anecdote rather than measurement. AgentForge removes the manual burden by orchestrating a fully automated improvement loop:

```
parse → generate tests → run → trace → score → optimize → gate → promote
```

Humans set the quality bar. The platform handles the repetitive evaluation and iteration work.

### Core Features (MVP)

| Feature | Description |
|---------|-------------|
| **F-01 Agent Loader** | Parses YAML/JSON/Markdown/Copilot `.agent.md` agent files, validates against schema, SHA-based version store |
| **F-02 Scenario Generator** | Generates N test scenarios via schema-derived, adversarial, and domain-seeded strategies |
| **F-03 Agent Runner** | Parallel execution with full trace capture, retry logic, and token usage tracking |
| **F-04 Trace Scorer** | Six-dimension weighted scoring via deterministic assertions + LLM-as-judge |
| **F-05 Optimizer** | Generates 5–20 candidate agent variants per cycle using mutation strategies |
| **F-06 Gatekeeper** | Three-gate promotion logic: score gate + regression gate + stability gate |
| **F-07 REST API** | Axum-based API with endpoints for agents, runs, diffs, and results |
| **F-08 CLI** | `agentforge run`, `diff`, `promote` commands with GitHub Actions support |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AGENTFORGE PLATFORM                          │
│                                                                      │
│  INPUT: Agent File (YAML / JSON / MD)                                │
│         system_prompt · tools[] · output_schema                      │
│         constraints[] · model  · sampling_config                     │
│                       │                                              │
│                       ▼                                              │
│  ┌────────────────────────────────┐                                  │
│  │  F-01: AGENT LOADER            │                                  │
│  │  Parser → Schema Validator     │                                  │
│  │  → Version Store (SHA-based)   │                                  │
│  └────────────────┬───────────────┘                                  │
│                   ▼                                                  │
│  ┌───────────────────────────────┐                                   │
│  │  F-02: SCENARIO GENERATOR     │                                   │
│  │  Schema-derived (50%)         │                                   │
│  │  Adversarial    (30%)         │                                   │
│  │  Domain-seeded  (20%)         │                                   │
│  └────────────────┬──────────────┘                                   │
│                   ▼                                                  │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  F-03: AGENT RUNNER (parallel workers, full trace capture) │      │
│  └────────────────────────────────┬───────────────────────────┘      │
│                                   ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  F-04: TRACE ANALYZER & SCORER                             │      │
│  │  Deterministic assertions + LLM-as-judge                   │      │
│  │  Weighted aggregate score + Failure cluster report         │      │
│  └─────────────────────────────────┬──────────────────────────┘      │
│                                    ▼                                 │
│  ┌──────────────────────────────────────┐                            │
│  │  F-05: OPTIMIZER                     │                            │
│  │  Prompt rewrite · Tool desc rewrite  │                            │
│  │  Schema tighten · Example inject     │                            │
│  │  → 5–20 Candidate Variants           │                            │
│  └──────────────────────┬───────────────┘                            │
│                         ▼                                            │
│  ┌─────────────────────────────────┐                                 │
│  │  F-06: PROMOTION GATEKEEPER     │                                 │
│  │  Score Gate (+3% over champion) │                                 │
│  │  Regression Gate (≥99% pass)    │                                 │
│  │  Stability Gate (3 seeds)       │                                 │
│  └─────────────────────┬───────────┘                                 │
│                        ▼                                             │
│  ┌───────────────────────────────────────┐                           │
│  │  PROMOTED AGENT FILE                  │                           │
│  │  (versioned, diffed, changelog)       │                           │
│  └───────────────────────────────────────┘                           │
│                                                                      │
│  F-07: REST API   │  F-08: CLI / GitHub Actions                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

This is a Cargo workspace with 16 crates:

```
agentforge/
├── Cargo.toml                  # Workspace root
├── Cargo.lock
├── docker-compose.yml          # PostgreSQL for local dev
├── Dockerfile
├── .env.example                # Environment variable template
├── migrations/                 # SQLx database migrations
│   ├── 001_agent_versions.sql
│   ├── 002_eval_runs.sql
│   ├── 003_scenarios.sql
│   ├── 004_traces.sql
│   ├── 005_shadow_runs.sql
│   ├── 006_finetune_exports.sql
│   ├── 007_benchmarks.sql
│   └── 008_trace_cost.sql
├── fixtures/
│   ├── customer-support-agent.yaml     # Native YAML example
│   └── agentforge-evaluator.agent.md   # Copilot .agent.md example
└── crates/
    ├── agentforge-core/        # Shared types, errors, traits (AgentFile, EvalRun, Trace, Scenario…)
    ├── agentforge-parser/      # Agent file parsing (YAML, JSON, Markdown/Copilot frontmatter)
    ├── agentforge-scenarios/   # Scenario generation (schema-derived, adversarial, domain-seeded)
    ├── agentforge-runner/      # Parallel agent execution + full trace capture
    ├── agentforge-scorer/      # Deterministic assertions + LLM-as-judge scoring
    ├── agentforge-optimizer/   # Variant generation + self-improvement loop
    ├── agentforge-gatekeeper/  # Three-gate promotion logic
    ├── agentforge-db/          # PostgreSQL repository layer (SQLx)
    ├── agentforge-api/         # REST API (Axum 0.8)
    ├── agentforge-cli/         # CLI binary (Clap 4)
    ├── agentforge-benchmarks/  # Standard benchmark comparison suite
    ├── agentforge-finetune/    # Fine-tune dataset exporter (JSONL)
    ├── agentforge-multiagent/  # Multi-agent composition testing
    ├── agentforge-observability/ # OTLP trace export hooks
    ├── agentforge-online-eval/ # Shadow-mode real-traffic comparison
    └── agentforge-redteam/     # Adversarial safety red-team probes
```

---

## Quick Start

### Prerequisites

- Rust 1.83+ (install via [rustup](https://rustup.rs))
- Docker + Docker Compose
- OpenAI, Anthropic, or NVIDIA NIM API key
- Node.js 20+ and npm (only required if developing the `web/` dashboard locally; production deployment uses the pre-built Docker image)

### 1. Clone and start infrastructure

```bash
git clone https://github.com/bhavinkotak/agentforge.git
cd agentforge

# Start PostgreSQL and Redis
docker-compose up -d

# Copy and configure environment
cp .env.example .env
# Edit .env — add at minimum one LLM API key:
# OpenAI:   OPENAI_API_KEY=sk-...
# Anthropic: ANTHROPIC_API_KEY=sk-ant-...
# NVIDIA NIM (free tier): NVIDIA_API_KEY=nvapi-...
```

### 2. Run database migrations

```bash
export DATABASE_URL="postgres://agentforge:agentforge@localhost:5432/agentforge"
cargo install sqlx-cli --no-default-features --features postgres
sqlx migrate run
```

### 3. Build and run the API server

```bash
cargo build --release
DATABASE_URL=$DATABASE_URL ./target/release/agentforge-api
# Server starts on http://127.0.0.1:8080
```

### 4. Run your first eval via CLI

```bash
# Run a full evaluation cycle
./target/release/agentforge run \
  --agent fixtures/customer-support-agent.yaml \
  --scenarios 50

# Show a scorecard diff between two versions
./target/release/agentforge diff <version-id-1> <version-id-2>

# Promote the winning version (pass the run-id returned by `agentforge run`)
./target/release/agentforge promote <run-id>
```

---

## Agent File Format

AgentForge accepts agent files in the following formats:
- **AgentForge native YAML** (recommended)
- **GitHub Copilot `.agent.md`** — YAML frontmatter + Markdown system prompt body
- OpenAI Assistants API JSON
- Anthropic Claude system prompt + tool block JSON
- LangChain / LangGraph agent YAML
- CrewAI agent definition YAML

### GitHub Copilot `.agent.md` Format

Compatible with agents from [github/awesome-copilot](https://github.com/github/awesome-copilot/tree/main/agents). The frontmatter holds metadata; the Markdown body becomes the system prompt.

```markdown
---
name: 'Code Review Expert'
description: 'Specialist in reviewing code for security and maintainability'
model: GPT-4.1
tools: ['read', 'search/codebase', 'github/*']
---

# Code Review Expert

You are an expert code reviewer specializing in security, performance,
and maintainability.

## Review Focus Areas

- **Security**: Check for injection vulnerabilities and data exposure
- **Performance**: Identify N+1 queries and unnecessary allocations
- **Maintainability**: Evaluate clarity and SOLID principles
```

**Frontmatter fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Agent display name |
| `description` | string | Short description (stored in metadata) |
| `model` | string | LLM model ID — infers provider from name (e.g. `GPT-4.1` → OpenAI, `claude-*` → Anthropic) |
| `tools` | string[] | Capability references like `"github/*"`, `"read"`, `"context7/*"` |
| `argument-hint` | string | Hint for the argument the agent expects (stored in metadata) |
| `mcp-servers` | object | MCP server configurations (stored in metadata) |

Copilot tool capability references are mapped to `ToolDefinition` entries so AgentForge can reason about them during scenario generation and scoring.

### AgentForge Native YAML Schema

```yaml
# agent.yaml — AgentForge native schema v1
agentforge_schema_version: "1"
name: "customer-support-agent"
version: "2.1.0"

model:
  provider: openai          # openai | anthropic | ollama | bedrock
  model_id: gpt-4o
  temperature: 0.2
  max_tokens: 2048

system_prompt: |
  You are a helpful customer support agent for Acme Corp.
  Always greet the user by name if known.
  Never share pricing without verifying entitlement first.

tools:
  - name: get_order_status
    description: "Retrieve status of a customer order by order ID."
    parameters:
      type: object
      properties:
        order_id:
          type: string
          description: "The order identifier, format: ORD-XXXXXXXX"
      required: [order_id]

output_schema:
  type: object
  properties:
    response:
      type: string
    action_taken:
      type: string
      enum: [escalate, resolved, needs_followup, no_action]
    confidence:
      type: number
      minimum: 0
      maximum: 1
  required: [response, action_taken]

constraints:
  - "Never mention competitor products."
  - "Do not provide refunds without running check_refund_eligibility first."
  - "Always confirm order ID before calling get_order_status."

eval_hints:
  domain: customer_support
  typical_turns: 3
  critical_tools: [get_order_status, check_refund_eligibility]
  pass_threshold: 0.85    # minimum aggregate score to promote
  scenario_count: 200
```

---

## CLI Usage

```
agentforge <COMMAND> [OPTIONS]

Commands:
  run      Run a full eval cycle (parse → generate → run → score → optimize → gate)
  diff     Show scorecard diff between two agent versions
             Usage: agentforge diff <version-id-1> <version-id-2>
  promote  Promote a candidate version to champion
             Usage: agentforge promote <run-id>
  help     Print help

Options for `run`:
  --agent <FILE>               Path to agent YAML/JSON file (required)
  --scenarios <N>              Number of scenarios to generate (default: 100)
  --concurrency <N>            Parallel workers (default: 10)
  --seed <N>                   Random seed for reproducibility (default: 42)
  --provider <NAME>            Agent LLM provider: openai | anthropic | nvidia | ollama | bedrock (default: openai)
                               (ollama and bedrock require a self-hosted runner with local infrastructure)
  --judge-provider <NAME>      Judge LLM provider (must differ from --provider; default: anthropic)
  --threshold <F>              Pass threshold 0.0–1.0 (default: 0.85)
  --output-json <FILE>         Write full scorecard JSON to FILE (used by the GitHub Action)
  --dry-run                    Validate agent file and preview scenario count; no LLM calls made
  --max-cost <USD>             Abort if estimated cost exceeds USD (e.g. --max-cost 5.00); 0 = no cap
  --agent-format <FMT>         Override format detection: native_yaml | openai_json | anthropic_json | langchain_yaml | crewai_yaml | copilot_agent_md
  --weight-task <F>            Override task-completion weight (default: 0.35)
  --weight-tool <F>            Override tool-selection weight (default: 0.20)
  --weight-args <F>            Override argument-correctness weight (default: 0.20)
  --weight-schema <F>          Override schema-compliance weight (default: 0.15)
  --weight-instr <F>           Override instruction-adherence weight (default: 0.07)
  --weight-path <F>            Override path-efficiency weight (default: 0.03)
  --red-team                   Append adversarial red-team probes to standard scenarios
  --cost-optimize              After eval, recommend cheaper model alternatives
  --watch                      Re-run the evaluation automatically when the agent file is saved.
                               Polls for file changes every 500 ms. Press Ctrl-C to stop.

Exit codes:
  0  — All gates passed, version promoted (or no promotion needed)
  1  — Gatekeeper blocked promotion
  2  — Error (parse failure, DB connection, etc.)
```

---

## REST API

The API server runs on `http://0.0.0.0:8080` by default.

An **OpenAPI 3.1 spec** is available at [docs/openapi.yaml](docs/openapi.yaml) and can be loaded into Swagger UI, Postman, or any OpenAPI-compatible tool.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/agents` | List all registered agent versions (paginated: `?limit=50&offset=0`) |
| `POST` | `/api/v1/agents` | Upload and register a new agent version |
| `GET` | `/api/v1/agents/:id` | Get agent version by ID |
| `PATCH` | `/api/v1/agents/:id` | Update agent metadata (`is_champion`, `changelog`) |
| `DELETE` | `/api/v1/agents/:id` | Delete an agent version (blocked if it is the current champion) |
| `GET` | `/api/v1/agents/:id/scenarios` | List generated test scenarios for an agent version (paginated: `?limit=50`, max 500) |
| `GET` | `/api/v1/runs` | List all eval runs (paginated: `?limit=50&offset=0`) |
| `POST` | `/api/v1/runs` | Start a new eval run (rate-limited by `AGENTFORGE_MAX_CONCURRENT_RUNS`) |
| `GET` | `/api/v1/runs/:id` | Get run status and results |
| `DELETE` | `/api/v1/runs/:id` | Cancel a pending/running eval run (sets status → `cancelled`) |
| `GET` | `/api/v1/runs/:id/scorecard` | Full scorecard with per-dimension scores and failure clusters |
| `GET` | `/api/v1/runs/:id/traces` | List traces for a run (paginated: `?limit=100&offset=0`, max 500) |
| `GET` | `/api/v1/runs/:id/progress` | **Server-Sent Events** stream of live run progress (emits every ~2 s until terminal) |
| `GET` | `/api/v1/diff` | Scorecard diff between two versions (`?v1=<uuid>&v2=<uuid>`) |
| `POST` | `/api/v1/promote/:run_id` | Promote version to champion (runs all three gatekeeper gates) |
| `GET` | `/health` | Liveness probe — exempt from API key authentication |

> **Concurrency limit on `POST /runs`:** to prevent accidental LLM cost floods, the server rejects new
> eval-run requests with HTTP 429 when `AGENTFORGE_MAX_CONCURRENT_RUNS` active background tasks are
> already running (default: `10`). For high-throughput CI, raise this value in your deployment env.
> For per-client rate limiting, place a reverse proxy (nginx / Cloudflare) in front of the API.

> **Scenario count limit on `POST /runs`:** `scenario_count` must not exceed `AGENTFORGE_MAX_SCENARIOS`
> (default: `2000`). Requests that exceed this value are rejected with HTTP 400.

> **`auto_optimize` on `POST /runs`:** set `"auto_optimize": true` to enable the self-improvement loop.
> After the eval completes, if the aggregate score is below `0.85`, the optimizer generates up to 5
> candidate variants (prompt rewrites, tool-description rewrites, example injections, constraint
> tightenings), quick-evaluates each on a 10-scenario subset, and saves the best-performing variant
> as a new agent version in the database. The parent SHA is recorded for full lineage tracking.

> **Authentication:** set `AGENTFORGE_API_KEY` to require a Bearer token on all `/api/v1/*` endpoints.
> Requests without a valid `Authorization: Bearer <key>` header will receive HTTP 401. The `/health`
> endpoint is always unauthenticated. When the env var is unset the server runs in unauthenticated
> development mode.

### Example: Start an eval run

```bash
# Upload agent file (YAML or JSON; Content-Type should match the file format)
curl -X POST http://localhost:8080/api/v1/agents \
  -H "Content-Type: application/yaml" \
  --data-binary @fixtures/customer-support-agent.yaml

# Start eval run (all fields except agent_id are optional)
# Add -H "Authorization: Bearer $AGENTFORGE_API_KEY" when authentication is enabled
curl -X POST http://localhost:8080/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "550e8400-e29b-41d4-a716-446655440000",
    "scenario_count": 100,
    "concurrency": 10,
    "seed": 42,
    "threshold": 0.85,
    "provider": "openai",
    "judge_provider": "anthropic",
    "auto_optimize": true
  }'

# Poll for run results (replace with the run UUID returned above)
curl http://localhost:8080/api/v1/runs/7dc53df0-c5fa-4f6c-b6b5-8d2d19afe5b1
```

---

## Scoring Dimensions

Every execution trace is scored across six dimensions:

| Dimension | Weight | Scoring Method | What is Measured |
|-----------|--------|---------------|-----------------|
| Task completion | 35% | Deterministic + LLM judge | Did the agent achieve the stated goal? |
| Tool selection accuracy | 20% | Exact match | Were the correct tools called? |
| Argument correctness | 20% | JSON schema + semantic | Were tool arguments valid and semantically correct? |
| Output schema compliance | 15% | JSON schema strict | Does output match the declared schema? |
| Instruction adherence | 7% | LLM judge with rubric | Did the agent follow all behavioral constraints? |
| Path efficiency | 3% | Step count vs. optimal | Was the shortest valid path taken? |

Weights are configurable via environment variables (`AGENTFORGE_WEIGHT_TASK`, `AGENTFORGE_WEIGHT_TOOL`, `AGENTFORGE_WEIGHT_ARGS`, `AGENTFORGE_WEIGHT_SCHEMA`, `AGENTFORGE_WEIGHT_INSTR`, `AGENTFORGE_WEIGHT_PATH`) or via per-run CLI flags (`--weight-task`, `--weight-tool`, etc.). The judge LLM **must use a different provider from the agent** to prevent circular bias — enforcement is at the provider level (e.g., `openai` vs `anthropic`), not the individual model ID. The API returns HTTP 400 if `provider` and `judge_provider` are the same string.

### Failure Clusters

Traces are automatically grouped into failure clusters:

| Cluster | Meaning |
|---------|----------|
| `wrong_tool` | Called an incorrect or unnecessary tool |
| `hallucinated_argument` | Passed a fabricated or invalid argument value |
| `looping` | Repeated the same tool call without progress |
| `premature_stop` | Ended the conversation before completing the task |
| `schema_violation` | Output did not match the declared schema |
| `constraint_breach` | Violated a behavioural constraint |
| `api_error` | Infrastructure failure (rate limit, 5xx, timeout) — not an agent quality issue |
| `no_failure` | Trace passed — no failure to classify |

---

## Promotion Gatekeeper

A candidate variant must clear **all three gates** to be promoted:

1. **Score Gate** — Aggregate score must exceed the current champion by at least `+3%` (configurable via `AGENTFORGE_SCORE_GATE_DELTA`).

2. **Regression Gate** — Must pass ≥ 99% of the scenarios the current champion passes (configurable via `AGENTFORGE_REGRESSION_GATE_THRESHOLD`). Prevents "robbing Peter to pay Paul" improvements.

3. **Stability Gate** — Must be evaluated on at least 3 independent random seeds before comparison, to account for LLM non-determinism (configurable via `AGENTFORGE_STABILITY_SEEDS`).

If multiple candidates pass all gates, the one with the highest aggregate score is promoted. Promotion creates a new versioned agent file with an auto-generated changelog entry.

> **First run (no champion):** When no champion exists yet, all three gates are automatically **waived** and the candidate is promoted unconditionally. This bootstraps the system on the first evaluation.

---

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |
| `REDIS_URL` | `redis://localhost:6379` | Redis for caching |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `NVIDIA_API_KEY` | — | NVIDIA NIM API key (`nvapi-…`) |
| `AGENTFORGE_NVIDIA_MODEL` | `mistralai/mistral-small-4-119b-2603` | NVIDIA NIM model to use for the agent |
| `AGENTFORGE_JUDGE_BASE_URL` | _(provider default)_ | Override the judge LLM base URL (useful for OpenAI-compatible endpoints) |
| `AGENTFORGE_HOST` | `127.0.0.1` | API server bind address |
| `AGENTFORGE_PORT` | `8080` | API server port |
| `AGENTFORGE_LOG_LEVEL` | `info` | Log level (trace/debug/info/warn/error) |
| `AGENTFORGE_MAX_CONCURRENT_RUNS` | `10` | Max simultaneous background eval runs (HTTP 429 when exceeded) |
| `AGENTFORGE_API_KEY` | — | Bearer token for API authentication. When set, all `/api/v1/*` endpoints require `Authorization: Bearer <key>`. Unset = unauthenticated dev mode |
| `AGENTFORGE_JUDGE_PROVIDER` | `openai` | LLM provider for the judge |
| `AGENTFORGE_JUDGE_MODEL` | `gpt-4o` | Judge model ID |
| `AGENTFORGE_DEFAULT_SCENARIOS` | `100` | Default scenario count per run |
| `AGENTFORGE_MAX_SCENARIOS` | `2000` | Maximum scenarios allowed per run (HTTP 400 when exceeded) |
| `AGENTFORGE_DEFAULT_CONCURRENCY` | `10` | Parallel worker count |
| `AGENTFORGE_DEFAULT_PASS_THRESHOLD` | `0.85` | Minimum score to pass a run |
| `AGENTFORGE_SCORE_GATE_DELTA` | `0.03` | Required score improvement to promote |
| `AGENTFORGE_REGRESSION_GATE_THRESHOLD` | `0.99` | Required pass-rate on champion scenarios |
| `AGENTFORGE_STABILITY_SEEDS` | `3` | Seeds required for stability gate |
| `AGENTFORGE_WEIGHT_TASK` | `0.35` | Task-completion scoring weight |
| `AGENTFORGE_WEIGHT_TOOL` | `0.20` | Tool-selection scoring weight |
| `AGENTFORGE_WEIGHT_ARGS` | `0.20` | Argument-correctness scoring weight |
| `AGENTFORGE_WEIGHT_SCHEMA` | `0.15` | Schema-compliance scoring weight |
| `AGENTFORGE_WEIGHT_INSTR` | `0.07` | Instruction-adherence scoring weight |
| `AGENTFORGE_WEIGHT_PATH` | `0.03` | Path-efficiency scoring weight |
| `AGENTFORGE_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Base URL for Ollama (OpenAI-compatible API) |
| `AGENTFORGE_NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Base URL for NVIDIA NIM |

> **`REDIS_URL` vs `AGENTFORGE_*`:** Redis uses the bare `REDIS_URL` key (compatible with most hosting platforms). All other app-level settings use the `AGENTFORGE_` prefix.

> **`AGENTFORGE_HOST` default:** The server binds to `127.0.0.1` by default for security (localhost only). Set to `0.0.0.0` to accept external connections — docker-compose overrides this automatically for container networking.

---

## Running Tests

```bash
# Start PostgreSQL first
docker-compose up -d postgres

# Run all tests
DATABASE_URL="postgres://agentforge:agentforge@localhost:5432/agentforge" \
  cargo test --workspace

# Run tests for a specific crate
cargo test -p agentforge-scorer
cargo test -p agentforge-runner
cargo test -p agentforge-gatekeeper

# Run with output
cargo test --workspace -- --nocapture
```

The test suite covers:
- Agent file parsing (all 6 formats)
- Scenario generation (schema-derived, adversarial, domain-seeded)
- Runner execution with mocked LLM
- Scoring logic (all 6 dimensions)
- Optimizer variant generation
- Gatekeeper promotion logic
- REST API integration tests
- Database repository tests

---

## GitHub Actions Marketplace

[![GitHub Marketplace](https://img.shields.io/badge/GitHub%20Actions-Marketplace-blue?logo=github)](https://github.com/marketplace/actions/agentforge-eval)

AgentForge is published as a reusable GitHub Action. No Rust toolchain, database, or build step required in your workflow — the action downloads a pre-built binary from the latest release automatically.

```yaml
- uses: bhavinkotak/agentforge@v1
  with:
    agent_file: './agents/my-agent.yaml'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    DATABASE_URL: ${{ secrets.AGENTFORGE_DATABASE_URL }}
```

### Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `agent_file` | **yes** | — | Path to the agent file (YAML, JSON, or `.agent.md`) |
| `scenarios` | no | `100` | Number of test scenarios to generate |
| `concurrency` | no | `10` | Parallel workers for running scenarios |
| `seed` | no | `42` | Random seed for reproducible scenario generation |
| `threshold` | no | `0.85` | Minimum aggregate score to gate promotion (0.0–1.0) |
| `provider` | no | `openai` | LLM provider for the agent under test: `openai` \| `anthropic` \| `nvidia` \| `ollama` \| `bedrock` (ollama and bedrock require a self-hosted runner) |
| `judge_provider` | no | `anthropic` | Judge LLM provider (must differ from `provider` at the provider level): `openai` \| `anthropic` \| `nvidia` \| `ollama` \| `bedrock` |
| `version` | no | _(action ref)_ | Specific AgentForge release to use (e.g. `v1.2.3`). Defaults to the version of this action. |

### Outputs

| Output | Description |
|--------|-------------|
| `pass_rate` | Aggregate pass rate across all evaluated scenarios (0.0–1.0) |
| `aggregate_score` | Weighted aggregate score across all six dimensions (0.0–1.0) |
| `promoted` | Whether the agent was promoted to champion (`"true"` \| `"false"`) |
| `scorecard_path` | Path to the JSON scorecard artifact |
| `run_id` | AgentForge eval run UUID |

### Use Cases

#### Block a merge when agent quality drops

Run a full eval cycle on every PR that touches agent files. Fail the check if the score falls below threshold — preventing regressions from being merged.

```yaml
name: Agent Quality Gate
on:
  pull_request:
    paths: ['agents/**', '*.agent.md']

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: bhavinkotak/agentforge@v1
        id: eval
        with:
          agent_file: './agents/customer-support-agent.yaml'
          scenarios: '200'
          threshold: '0.85'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          DATABASE_URL: ${{ secrets.AGENTFORGE_DATABASE_URL }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}   # enables PR comment
      - name: Report scorecard
        if: always()
        run: |
          echo "Pass rate: ${{ steps.eval.outputs.pass_rate }}"
          echo "Aggregate score: ${{ steps.eval.outputs.aggregate_score }}"
          echo "Promoted: ${{ steps.eval.outputs.promoted }}"
```

#### Nightly improvement loop

Run AgentForge on a schedule to continuously generate, evaluate, and auto-promote improved agent variants.

```yaml
name: Nightly Agent Improvement
on:
  schedule:
    - cron: '0 2 * * *'   # 02:00 UTC every night

jobs:
  improve:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: bhavinkotak/agentforge@v1
        with:
          agent_file: './agents/customer-support-agent.yaml'
          scenarios: '500'
          concurrency: '20'
          threshold: '0.88'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          DATABASE_URL: ${{ secrets.AGENTFORGE_DATABASE_URL }}
```

#### Evaluate a GitHub Copilot `.agent.md` file

AgentForge natively parses Copilot agent files — just point `agent_file` at the `.agent.md`.

```yaml
- uses: bhavinkotak/agentforge@v1
  with:
    agent_file: '.github/agents/code-review.agent.md'
    scenarios: '100'
    provider: 'openai'
    judge_provider: 'anthropic'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    DATABASE_URL: ${{ secrets.AGENTFORGE_DATABASE_URL }}
```

> **Tip:** Pin to a specific version for reproducibility: `bhavinkotak/agentforge@v1.2.3`

---

## CI/CD Integration

### Self-hosted / custom CI

If you prefer to build from source (e.g., air-gapped environments), use the CLI directly:

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation

on:
  push:
    paths: ['agents/**']
  pull_request:
    paths: ['agents/**']

jobs:
  evaluate:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: agentforge
          POSTGRES_PASSWORD: agentforge
          POSTGRES_DB: agentforge
        ports:
          - 5432:5432
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Build AgentForge CLI
        run: cargo build --release -p agentforge-cli

      - name: Run Migrations
        env:
          DATABASE_URL: postgres://agentforge:agentforge@localhost:5432/agentforge
        run: cargo sqlx migrate run

      - name: Run AgentForge Evaluation
        env:
          DATABASE_URL: postgres://agentforge:agentforge@localhost:5432/agentforge
          REDIS_URL: redis://localhost:6379
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          ./target/release/agentforge run \
            --agent ./agents/customer-support-agent.yaml \
            --scenarios 200 \
            --threshold 0.85
```

Exit codes: `0` = passed/promoted, `1` = gatekeeper blocked, `2` = error.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the full process. In short:

1. **Open an issue first** for significant changes so we can discuss the approach.
2. **Fork and branch** — create a feature branch from `main`.
3. **Follow code conventions** — `cargo fmt` and `cargo clippy --all-targets` must pass.
4. **Add tests** — all new behaviour must be covered.
5. **Open a PR** — all PRs require approval from the project maintainer ([@bhavinkotak](https://github.com/bhavinkotak)) before merging.

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev environment setup, commit message conventions, and the full review checklist.

---

## Technical Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | **Rust** | Memory safety, zero-cost abstractions, deterministic performance |
| API framework | Axum 0.8 | Async, ergonomic, tower-compatible middleware |
| Database | PostgreSQL 16 + SQLx 0.8 | Relational integrity + offline query checking |
| Caching | Redis (deadpool-redis) | Run state, rate limit tracking |
| LLM clients | reqwest 0.12 (rustls) | Async HTTP with TLS, no native deps |
| CLI | Clap 4 (derive) | Zero-boilerplate argument parsing |
| Async runtime | Tokio 1 (full) | Production async runtime |
| Serialization | serde + serde_json + serde_yaml | Full format support |
| Testing | tokio-test, mockall 0.14, wiremock 0.6 | Async mocks without external services |
| Observability | tracing 0.1 + tracing-subscriber | Structured logs, span context |

---

## Roadmap

### v2 (Post-MVP)

| Feature | Description |
|---------|-------------|
| Online eval | ✅ Implemented — shadow-mode real traffic comparison via `POST /api/v1/shadow-runs` |
| Fine-tune exporter | ✅ Implemented — export trace pairs as JSONL via `POST /api/v1/exports/finetune` |
| Multi-agent testing | ✅ Implemented — `agentforge-multiagent` crate for composed agent teams |
| Red-teaming mode | ✅ Implemented — adversarial safety probes via `--red-team` CLI flag |
| Benchmark comparison | ✅ Implemented — compare against standard suites via `POST /api/v1/benchmarks` |
| Observability hooks | ✅ Implemented — OTLP trace export via `OTEL_EXPORTER_OTLP_ENDPOINT` |
| Cost optimizer | ✅ Implemented — model downgrade recommendations via `--cost-optimize` CLI flag |

> **Web dashboard** (`web/`) is already included in this repo and served via the Docker Compose stack on port 3000.

---

## Privacy

AgentForge is a self-hosted platform. It sends **no telemetry, analytics, or usage data** to any external service. All evaluation data, agent files, and traces remain within your own infrastructure.

---

## License

MIT
