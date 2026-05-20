# AgentForge — Product Requirements Document & Architecture

**Version:** 1.0  
**Date:** April 26, 2026  
**Author:** Product / Engineering  
**Status:** Draft — Ready for Engineering Review

---

## Executive Summary

AgentForge is a self-improving AI agent optimization platform. The sole required input is an **agent file** — a declarative specification describing a conversational or agentic AI's system prompt, tools, behavior constraints, output schemas, and metadata. From that single artifact, AgentForge autonomously generates test scenarios, runs the agent, scores every execution trace, and iterates on the agent specification until it converges on a measurably better version. The product removes the traditional human-in-the-loop for routine tuning while maintaining a governed promotion gate so that only statistically validated improvements reach production.

---

## Problem Statement

AI agent development has a painful quality gap: teams ship prompts, tool definitions, and behavioral specs with little systematic testing, and improvements are made based on anecdote rather than measurement. [cite:11]

Current evaluation approaches require significant manual effort to:
- Write test cases by hand.
- Define scoring rubrics per task type.
- Interpret multi-step execution traces.
- Decide which prompt changes are genuine improvements vs. regressions. [cite:7][cite:13]

This manual burden means most teams test shallow happy-path scenarios rather than the full distribution of edge cases a deployed agent will encounter. [cite:15] The result is degraded quality in production that could have been caught systematically. [cite:11]

---

## Product Vision

> **One file in. A better agent out.**

AgentForge treats an agent file as the source of truth and orchestrates an autonomous improvement loop: parse → generate tests → run → trace → score → optimize → gate → promote. Humans set the quality bar and review failure clusters; the platform handles the repetitive evaluation and iteration work. [cite:7][cite:8]

---

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| Reduce prompt tuning time | Hours per agent version | < 1 hour (vs. 8–40 hours manually) |
| Improve agent task completion rate | End-to-end success % across eval suite | +15% over baseline in v1 |
| Detect regressions automatically | Regression catch rate | > 95% of breaking changes caught before promotion |
| Enable non-ML engineers to optimize agents | % of users without ML background shipping improvements | > 70% |
| Support all major agent formats | Agent file format coverage | OpenAI Assistants, LangChain, LangGraph, CrewAI, Claude, custom JSON/YAML |

---

## Target Users

### Primary: DevOps / Platform Engineers (Bhavin persona)
Engineers who own the CI/CD lifecycle for AI products — they want AgentForge to run as a step in a GitHub Actions pipeline, receive the agent file on each commit, and gate promotion the same way unit tests gate code merges. Comfort level: advanced; preference: YAML config, CLI, GitHub Actions integration.

### Secondary: AI Application Developers
Developers building on top of foundation models who iterate frequently on system prompts and tool definitions. They want a web UI to upload an agent file and see a scorecard in minutes.

### Tertiary: ML Engineers doing fine-tuning
Engineers who have enough failure data to run supervised fine-tuning. AgentForge feeds them labeled trace data automatically as the feedback loop matures.

---

## Feature Requirements

### MVP (v1) — Core Loop

#### F-01: Agent File Loader
- Accept a single file upload (YAML, JSON, or Markdown frontmatter format).
- Parse and extract: system prompt, tool definitions, expected output schema, behavioral constraints, model identifier, temperature/sampling config, memory settings.
- Validate the file against a schema registry; surface lint errors before any eval runs.
- Store parsed artifact as a versioned object with SHA-based content addressing.

**Supported input formats (v1):**
- OpenAI Assistants API JSON
- Anthropic Claude system prompt + tool block JSON
- LangChain/LangGraph agent YAML
- CrewAI agent definition YAML
- Generic `agent.yaml` (AgentForge native schema)

#### F-02: Scenario Generator
- Given the parsed agent file, generate N test scenarios (default: 100, configurable up to 2,000).
- Scenario generation strategy:
  1. **Schema-derived**: construct inputs that exercise every tool and output field.
  2. **Adversarial**: probe edge cases — empty inputs, contradictory instructions, tool call ordering failures, long context, multi-turn ambiguity.
  3. **Domain-seeded**: if the system prompt contains domain keywords, generate domain-relevant tasks automatically.
- Each scenario contains: `input`, `expected_tool_calls`, `expected_output_schema`, `pass_criteria`, `difficulty_tier` (easy / medium / hard / edge).
- Scenarios are stored, versioned, and reusable across agent versions.

#### F-03: Agent Runner
- Execute the agent against each scenario in an isolated sandbox.
- Capture full execution traces: every LLM call, every tool invocation, arguments, return values, intermediate reasoning, latency, token usage, and final output.
- Support parallel execution (configurable concurrency, default: 10 workers).
- Retry transient failures up to 3 times; mark persistent failures as `error` state.
- Emit structured trace logs per run (JSONL format).

#### F-04: Trace Analyzer & Scorer
Score each run across six dimensions: [cite:7]

| Dimension | What is Measured | Scoring Method |
|-----------|-----------------|----------------|
| Task completion | Did the agent achieve the goal? | Deterministic check + LLM judge |
| Tool selection accuracy | Were the right tools called? | Exact match against expected set |
| Argument correctness | Were tool arguments valid and correct? | JSON schema validation + semantic check |
| Path efficiency | Was the shortest valid path taken? | Step count vs. optimal count |
| Output schema compliance | Does output match the declared schema? | JSON schema strict validation |
| Instruction adherence | Did the agent follow behavioral constraints? | LLM-as-judge with rubric |

- Compute an **aggregate pass rate** and a **weighted score** per run.
- Cluster failures by root cause: wrong tool, hallucinated argument, looping, premature stop, schema violation, constraint breach.
- Surface a failure cluster report after every eval run.

#### F-05: Optimizer
- Generate **5–20 candidate variants** of the agent file per optimization cycle.
- Variant mutation strategies:
  - Prompt rewriting (clarity, specificity, constraint tightening).
  - Tool description rewriting (argument names, descriptions, examples).
  - Output schema tightening (stricter types, required fields).
  - Instruction ordering (move critical rules earlier in the system prompt).
  - Few-shot example injection (derive examples from passing traces).
- Each variant is a git-diff-style patch on the original agent file.

#### F-06: Promotion Gatekeeper
- A candidate variant must clear **all three gates** to be promoted:
  1. **Score gate**: aggregate score must exceed the current champion by a configurable threshold (default: +3%).
  2. **Regression gate**: must pass ≥ 99% of scenarios the champion already passes.
  3. **Stability gate**: must be run on at least 3 independent seeds before comparison to account for LLM non-determinism.
- If multiple candidates pass, the one with the highest aggregate score wins.
- Promotion creates a new versioned agent file with a changelog entry explaining what changed and by how much.

#### F-07: Leaderboard & Scorecard UI
- Dashboard showing: current champion version, score trend over versions, failure cluster distribution, token cost per run, latency distribution.
- Diff viewer: side-by-side view of two agent file versions with delta highlighting and score change annotations.
- Trace replay: step-through execution trace viewer for any run.
- Human review queue: failed traces that could not be automatically scored go here for labeling.

#### F-08: CLI + GitHub Actions Integration
- `agentforge run --agent ./agent.yaml --scenarios 200` — trigger a full eval run.
- `agentforge diff v3 v4` — show scorecard diff between two versions.
- `agentforge promote v4` — push the winning version to the production registry.
- GitHub Actions native action: `agentforge/eval-action@v1`.
- Exit code conventions: 0 = passed gates, 1 = failed gates, 2 = error.

---

### v2 Features (Post-MVP)

| Feature | Description |
|---------|-------------|
| **Online eval** | Shadow-mode real traffic comparison between current and candidate. [cite:13] |
| **Fine-tune exporter** | Export labeled trace pairs as JSONL for OpenAI, Anthropic, or Hugging Face fine-tuning. |
| **Multi-agent testing** | Test agent teams (CrewAI, LangGraph graphs) as a composed unit. [cite:8] |
| **Red-teaming mode** | Adversarial safety probing — jailbreak attempts, prompt injection, data leakage. [cite:8] |
| **Benchmark comparison** | Compare agent performance against public benchmarks (GAIA, AgentBench, WebArena). |
| **Observability hooks** | Export traces to Datadog, Grafana, LangSmith, or any OTLP-compatible backend. [cite:13] |
| **Cost optimizer** | Recommend model downgrades (e.g., GPT-4o → GPT-4o-mini) for scenarios where smaller models score equally. |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AGENTFORGE PLATFORM                          │
│                                                                      │
│  INPUT LAYER                                                         │
│  ┌─────────────────────────────────────────────────────┐            │
│  │  Agent File (YAML / JSON / MD)                      │            │
│  │  ── system_prompt  ── tools[]  ── output_schema     │            │
│  │  ── constraints[]  ── model    ── sampling_config   │            │
│  └───────────────────┬─────────────────────────────────┘            │
│                       │                                              │
│                       ▼                                              │
│  ┌────────────────────────────────┐                                  │
│  │   F-01: AGENT LOADER           │                                  │
│  │   Parser → Schema Validator    │                                  │
│  │   → Version Store (SHA-based)  │                                  │
│  └────────────────┬───────────────┘                                  │
│                   │                                                  │
│         ┌─────────▼──────────┐                                       │
│         │  F-02: SCENARIO    │                                       │
│         │  GENERATOR         │                                       │
│         │  Schema-derived    │                                       │
│         │  Adversarial       │                                       │
│         │  Domain-seeded     │                                       │
│         │  ─────────────     │                                       │
│         │  Scenario Store    │                                       │
│         └─────────┬──────────┘                                       │
│                   │  N scenarios                                     │
│                   ▼                                                  │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │   F-03: AGENT RUNNER (isolated sandbox, parallel workers)  │      │
│  │                                                            │      │
│  │   Worker 1  Worker 2  Worker 3  ...  Worker N              │      │
│  │      │          │        │               │                 │      │
│  │      └──────────┴────────┴───────────────┘                 │      │
│  │                         │                                  │      │
│  │            Full Execution Traces (JSONL)                   │      │
│  └─────────────────────────┬──────────────────────────────────┘      │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────┐             │
│  │   F-04: TRACE ANALYZER & SCORER                     │             │
│  │                                                     │             │
│  │   ┌─────────────┐   ┌──────────────┐               │             │
│  │   │ Deterministic│   │  LLM Judge   │               │             │
│  │   │  Assertions  │   │  (rubric)    │               │             │
│  │   └──────┬───────┘   └──────┬───────┘               │             │
│  │          └──────────────────┘                       │             │
│  │                    │                                │             │
│  │         Weighted Aggregate Score                    │             │
│  │         Failure Cluster Report                      │             │
│  │         Human Review Queue (ambiguous)              │             │
│  └─────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│              ┌─────────▼──────────┐                                  │
│              │  F-05: OPTIMIZER   │                                  │
│              │                    │                                  │
│              │  Prompt rewrite    │                                  │
│              │  Tool desc rewrite │                                  │
│              │  Schema tighten    │                                  │
│              │  Example inject    │                                  │
│              │  Instruction order │                                  │
│              │  ─────────────     │                                  │
│              │  5–20 Candidates   │                                  │
│              └─────────┬──────────┘                                  │
│                        │                                             │
│               ┌────────▼────────┐                                    │
│               │  F-06: GATEKEEPER│                                   │
│               │                  │                                   │
│               │  Score Gate      │                                   │
│               │  Regression Gate │                                   │
│               │  Stability Gate  │                                   │
│               └────────┬─────────┘                                   │
│                        │                                             │
│              ┌─────────▼──────────────┐                              │
│              │  PROMOTED AGENT FILE   │                              │
│              │  (versioned, diffed,   │                              │
│              │   changelog attached)  │                              │
│              └────────────────────────┘                              │
│                                                                      │
│  OUTPUT LAYER                                                        │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │  F-07: UI Dashboard   │  F-08: CLI / GitHub Actions      │        │
│  │  Leaderboard          │  agentforge run                  │        │
│  │  Trace Replay         │  agentforge diff                 │        │
│  │  Diff Viewer          │  agentforge promote              │        │
│  │  Human Review Queue   │  agentforge/eval-action@v1       │        │
│  └──────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Technical Stack

### Backend

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| API | Python FastAPI | Native async, OpenAPI schema auto-generation |
| Task queue | Celery + Redis | Parallel scenario workers, retry logic |
| Trace store | ClickHouse | Column-store ideal for JSONL trace analytics |
| Agent version store | PostgreSQL + pgvector | Relational metadata + vector similarity for scenario deduplication |
| Object store | S3-compatible (MinIO self-hosted, AWS S3 cloud) | Agent files, trace JSONL, eval artifacts |
| LLM judge | Configurable: GPT-4o, Claude 3.5 Sonnet, or local Ollama model | User controls judge model to prevent circular bias |
| Evaluation runner | Docker sandboxes per worker | Isolation prevents tool side effects from contaminating other runs |

### Frontend

| Layer | Technology |
|-------|-----------|
| Web UI | React + TypeScript (Vite) |
| Charts / Scorecard | Recharts |
| Trace viewer | Custom virtualized JSONL renderer |
| Diff viewer | Monaco Editor diff mode |

### CLI

| Component | Technology |
|-----------|-----------|
| CLI tool | Go (single binary, no runtime deps) |
| Config | `agentforge.yaml` in repo root |
| Auth | API key via env `AGENTFORGE_API_KEY` |

### Infrastructure

| Component | Choice |
|-----------|--------|
| Container orchestration | Kubernetes (Helm chart provided) |
| CI/CD integration | GitHub Actions native action |
| Secrets management | HashiCorp Vault / Azure Key Vault |
| Observability | OpenTelemetry → Grafana stack |

---

## Agent File Schema (AgentForge Native)

```yaml
# agent.yaml — AgentForge native schema (v1)
agentforge_schema_version: "1"
name: "customer-support-agent"
version: "2.1.0"
model:
  provider: openai            # openai | anthropic | ollama | bedrock
  model_id: gpt-4o
  temperature: 0.2
  max_tokens: 2048

system_prompt: |
  You are a helpful customer support agent for Acme Corp.
  Always greet the user by name if known.
  Never share pricing without verifying entitlement first.
  ...

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
  - "Do not provide refunds without running the check_refund_eligibility tool."
  - "Always confirm order ID before calling get_order_status."

eval_hints:
  domain: customer_support
  typical_turns: 3
  critical_tools: [get_order_status, check_refund_eligibility]
  pass_threshold: 0.85     # minimum aggregate score to promote
  scenario_count: 200
```

---

## Optimization Strategies (Detailed)

The optimizer applies mutations in an ordered priority based on known failure patterns: [cite:6][cite:7]

### Priority 1 — Prompt & Spec Clarity
The most impactful, cheapest change. Rewrite ambiguous instructions to be specific. Add explicit tool call ordering rules when traces show the wrong sequence. Insert negative examples for frequently hallucinated arguments.

### Priority 2 — Tool Description Quality
Many tool call failures trace back to vague parameter descriptions. [cite:7] The optimizer rewrites tool descriptions to include: type constraints, valid value examples, common misuse patterns to avoid, and explicit relationships between parameters.

### Priority 3 — Output Schema Tightening
Loose schemas allow the model to "pass" while producing structurally wrong output. The optimizer tightens required fields, adds enum constraints, and adjusts property descriptions to reduce schema violations.

### Priority 4 — Few-Shot Example Injection
After sufficient passing traces accumulate, the optimizer selects the highest-quality passing traces and injects them as in-context few-shot examples. This is particularly effective for complex multi-step tasks and domain-specific output formatting. [cite:13]

### Priority 5 — Model Selection
Only after prompts and tools are clean, the optimizer evaluates whether a smaller or cheaper model scores equivalently. A downgrade is only proposed if the candidate passes the regression gate.

### Priority 6 — Fine-Tuning Dataset Export
Once ≥ 500 labeled trace pairs exist, the platform exports a fine-tuning dataset. This path is reserved for teams that need to customize a base model beyond what prompt engineering achieves. [cite:15]

---

## Evaluation Framework

### Deterministic Assertions (no LLM required)

These checks run first, are cheap, and are definitive:

- JSON schema validation of final output.
- Tool call presence check (was required tool X called?).
- Argument type and format validation.
- Constraint keyword check (did the agent mention a forbidden term?).
- Conversation stop condition check (did the agent stop too early or loop?).

### LLM-as-Judge (for semantic correctness)

Used only when deterministic checks cannot score a dimension: [cite:7][cite:13]

- Task completion: LLM judge compares the agent's final response to the intended outcome.
- Instruction adherence: LLM judge evaluates behavioral constraint compliance against a natural-language rubric.
- Argument semantic correctness: Did the agent pass a sensible value even if the type was technically valid?

The judge model is **always different from the agent under test** to prevent circular bias. Judge model, temperature, and rubric are configurable.

### Scoring Weights (defaults, configurable)

| Dimension | Default Weight |
|-----------|---------------|
| Task completion | 35% |
| Tool selection accuracy | 20% |
| Argument correctness | 20% |
| Output schema compliance | 15% |
| Instruction adherence | 7% |
| Path efficiency | 3% |

---

## CI/CD Integration

### GitHub Actions workflow example

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation

on:
  push:
    paths:
      - 'agents/**'
  pull_request:
    paths:
      - 'agents/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run AgentForge Evaluation
        uses: agentforge/eval-action@v1
        with:
          agent_file: ./agents/customer-support-agent.yaml
          scenarios: 200
          pass_threshold: 0.85
          promote_on_pass: true
        env:
          AGENTFORGE_API_KEY: ${{ secrets.AGENTFORGE_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Comment scorecard on PR
        uses: agentforge/pr-comment-action@v1
        if: github.event_name == 'pull_request'
        with:
          scorecard_artifact: agentforge-scorecard.json
```

On pull request, AgentForge posts a scorecard comment showing the delta between base and head branch agent versions. On merge to main, the promoted agent is published to the agent registry.

---

## Data Model (Simplified)

```
AgentVersion
  id            UUID
  name          STRING
  version       SEMVER
  sha           STRING (content hash)
  file_content  JSONB
  created_at    TIMESTAMP
  promoted      BOOLEAN

EvalRun
  id            UUID
  agent_id      UUID → AgentVersion
  scenario_set  UUID
  status        ENUM(running, complete, error)
  aggregate_score FLOAT
  pass_rate     FLOAT
  started_at    TIMESTAMP
  completed_at  TIMESTAMP

Trace
  id            UUID
  run_id        UUID → EvalRun
  scenario_id   UUID
  steps         JSONB (full execution trace)
  scores        JSONB (per-dimension scores)
  failure_cluster STRING
  review_needed BOOLEAN

Scenario
  id            UUID
  agent_id      UUID → AgentVersion
  input         JSONB
  expected      JSONB
  difficulty    ENUM(easy, medium, hard, edge)
  domain        STRING
  auto_generated BOOLEAN
```

---

## Phased Roadmap

### Phase 1 — MVP (Weeks 1–8)

| Week | Milestone |
|------|-----------|
| 1–2 | Agent file parser + schema validator + version store |
| 2–3 | Scenario generator (schema-derived + adversarial) |
| 3–4 | Agent runner (Docker sandbox, parallel workers, trace capture) |
| 4–5 | Trace analyzer + deterministic scorer |
| 5–6 | LLM-as-judge scorer + failure cluster report |
| 6–7 | Optimizer (prompt + tool description mutations) |
| 7–8 | Gatekeeper + promotion flow + CLI + scorecard UI |

### Phase 2 — Production Hardening (Weeks 9–14)

| Week | Milestone |
|------|-----------|
| 9–10 | GitHub Actions native action + PR scorecard comments |
| 10–11 | Multi-format agent file support (LangChain, CrewAI, Claude) |
| 11–12 | Trace replay UI + diff viewer + leaderboard |
| 12–13 | Online / shadow eval mode |
| 13–14 | Fine-tune dataset exporter + cost optimizer module |

### Phase 3 — Enterprise (Weeks 15–20)

| Week | Milestone |
|------|-----------|
| 15–16 | Multi-agent graph testing (LangGraph, CrewAI teams) |
| 16–17 | Red-teaming / adversarial safety mode |
| 17–18 | RBAC, audit logs, SSO |
| 18–19 | Benchmark comparison (GAIA, AgentBench) |
| 19–20 | Self-hosted (Kubernetes Helm chart) + enterprise licensing |

---

## Open Questions & Risks

| Item | Risk Level | Mitigation |
|------|-----------|------------|
| LLM judge reliability | High | Use multiple judge models and take majority vote; allow human override |
| Non-determinism of agent outputs across seeds | Medium | Stability gate requires 3-seed average before promotion decision |
| Cost of running 200+ scenarios per eval cycle | Medium | Token budget controls, smaller model for deterministic-scoreable scenarios |
| Circular bias (agent and judge same model) | High | Enforced in config: judge model must differ from agent model |
| Agent format diversity | Medium | Start with 3 formats (OpenAI, LangChain, native YAML), expand by community request |
| Privacy of agent file contents | High | Encryption at rest, tenant isolation, no model training on customer agent files |

---

## Success Metrics at Launch

- **Agent score improvement**: ≥ 15% increase in aggregate score after one optimization cycle vs. the uploaded baseline, measured across 10 beta customer agents. [cite:7]
- **Time to first score**: < 10 minutes from file upload to first scorecard for a 200-scenario run.
- **Regression catch rate**: ≥ 95% of manually introduced regressions caught by the gatekeeper.
- **Developer adoption**: ≥ 80% of beta users integrate the GitHub Actions step within the first week.
- **Human review rate**: < 15% of traces require human review (i.e., ≥ 85% auto-scored). [cite:13]

