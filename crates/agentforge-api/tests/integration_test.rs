use uuid::Uuid;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use agentforge_core::{AgentFileFormat, ModelProvider};
use agentforge_parser::{parse_agent_file, to_agent_version, validate_agent_file};
use agentforge_scenarios::{generate_scenarios, ScenarioGeneratorConfig};

const SAMPLE_AGENT_YAML: &str = r#"
agentforge_schema_version: "1"
name: test-agent
version: "1.0.0"
model:
  provider: openai
  model_id: gpt-4o
  temperature: 0.2
system_prompt: "You are a helpful assistant. Answer questions concisely."
tools:
  - name: search
    description: "Search for information on the internet"
    parameters:
      type: object
      properties:
        query:
          type: string
          description: "The search query"
      required: [query]
output_schema:
  type: object
  properties:
    response:
      type: string
  required: [response]
constraints:
  - "Never provide harmful information"
eval_hints:
  scenario_count: 10
  pass_threshold: 0.75
"#;

const SAMPLE_COPILOT_AGENT_MD: &str = r#"---
name: 'Code Review Expert'
description: 'Specialist in reviewing code for security, performance, and maintainability'
model: GPT-4.1
tools: ['read', 'search/codebase', 'github/*']
---

# Code Review Expert

You are an expert code reviewer specializing in security, performance, and maintainability.

## Review Focus Areas

- **Security**: Check for injection vulnerabilities, authentication issues, and data exposure
- **Performance**: Identify N+1 queries, unnecessary allocations, and blocking operations
- **Maintainability**: Evaluate code clarity, test coverage, and adherence to SOLID principles

## Behavioral Constraints

- Always explain the reason behind each suggestion
- Provide concrete code examples when recommending changes
- Prioritize security issues above all others
"#;

// ─── Parser tests (no external dependencies) ────────────────────────────────

#[test]
fn parse_sample_agent_yaml() {
    let parsed = parse_agent_file(SAMPLE_AGENT_YAML).unwrap();
    assert_eq!(parsed.agent.name, "test-agent");
    assert_eq!(parsed.agent.version, "1.0.0");
    assert_eq!(parsed.agent.tools.len(), 1);
    assert!(!parsed.sha.is_empty());
}

#[test]
fn validate_sample_agent_passes() {
    let parsed = parse_agent_file(SAMPLE_AGENT_YAML).unwrap();
    let result = validate_agent_file(&parsed.agent);
    assert!(
        result.errors.is_empty(),
        "Unexpected errors: {:?}",
        result.errors
    );
}

#[test]
fn to_agent_version_produces_sha() {
    let parsed = parse_agent_file(SAMPLE_AGENT_YAML).unwrap();
    let version = to_agent_version(parsed);
    assert!(!version.sha.is_empty());
    assert_eq!(version.name, "test-agent");
}

#[test]
fn parse_copilot_agent_md_format() {
    let parsed = parse_agent_file(SAMPLE_COPILOT_AGENT_MD).unwrap();
    assert_eq!(parsed.format, AgentFileFormat::CopilotAgentMd);
    assert_eq!(parsed.agent.name, "Code Review Expert");
    assert_eq!(parsed.agent.model.model_id, "GPT-4.1");
    assert_eq!(parsed.agent.model.provider, ModelProvider::Openai);
    // System prompt is the Markdown body
    assert!(parsed.agent.system_prompt.contains("Code Review Expert"));
    assert!(parsed.agent.system_prompt.contains("Security"));
    // Tools are mapped from capability references
    assert_eq!(parsed.agent.tools.len(), 3);
    assert_eq!(parsed.agent.tools[0].name, "read");
    assert_eq!(parsed.agent.tools[1].name, "codebase");
    assert_eq!(parsed.agent.tools[2].name, "github");
}

#[test]
fn copilot_agent_md_description_in_metadata() {
    let parsed = parse_agent_file(SAMPLE_COPILOT_AGENT_MD).unwrap();
    let meta = parsed.agent.metadata.expect("should have metadata");
    assert_eq!(
        meta["description"].as_str().unwrap(),
        "Specialist in reviewing code for security, performance, and maintainability"
    );
}

#[test]
fn parse_copilot_agent_md_fixture_file() {
    let content = include_str!("../../../fixtures/agentforge-evaluator.agent.md");
    let parsed = parse_agent_file(content).unwrap();
    assert_eq!(parsed.format, AgentFileFormat::CopilotAgentMd);
    assert_eq!(parsed.agent.name, "AgentForge Evaluator");
    assert_eq!(parsed.agent.model.model_id, "gpt-4o");
    assert_eq!(parsed.agent.tools.len(), 4);
    assert!(!parsed.agent.system_prompt.is_empty());
}

// ─── Scenario generation (deterministic/adversarial) ──────────────────────

#[tokio::test]
async fn schema_derived_scenarios_generated() {
    let parsed = parse_agent_file(SAMPLE_AGENT_YAML).unwrap();
    let agent_id = Uuid::new_v4();

    let scenarios = generate_scenarios(
        &parsed.agent,
        &ScenarioGeneratorConfig {
            total_count: 10,
            agent_id,
            llm_api_key: None, // No LLM — falls back to heuristic
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert!(!scenarios.is_empty(), "Should generate at least 1 scenario");
    assert!(scenarios.len() <= 10);

    // Non-adversarial scenarios should have non-empty user messages;
    // adversarial "empty_input" scenarios are intentionally empty.
    use agentforge_core::ScenarioSource;
    for s in scenarios
        .iter()
        .filter(|s| s.source != ScenarioSource::Adversarial)
    {
        assert!(
            !s.input.user_message.is_empty(),
            "non-adversarial scenario has empty user_message"
        );
    }
}

#[tokio::test]
async fn adversarial_scenarios_include_edge_cases() {
    use agentforge_scenarios::adversarial::generate_adversarial_scenarios;
    let parsed = parse_agent_file(SAMPLE_AGENT_YAML).unwrap();
    let scenarios = generate_adversarial_scenarios(&parsed.agent, 10, Uuid::new_v4()).unwrap();
    assert!(
        scenarios.len() >= 5,
        "Expected at least 5 adversarial scenarios"
    );
}

// ─── Runner tests with mocked LLM ─────────────────────────────────────────

#[tokio::test]
async fn runner_completes_with_mocked_llm() {
    let mock_server = MockServer::start().await;

    // Mock OpenAI chat completions endpoint
    let mock_response = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"response\": \"I can help you with that search.\"}"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&mock_response))
        .mount(&mock_server)
        .await;

    let parsed = parse_agent_file(SAMPLE_AGENT_YAML).unwrap();
    let agent_id = Uuid::new_v4();

    // Generate a small set of scenarios
    let scenarios = generate_scenarios(
        &parsed.agent,
        &ScenarioGeneratorConfig {
            total_count: 3,
            agent_id,
            llm_api_key: None,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Create a runner pointing at the mock server
    use agentforge_runner::{AgentRunner, OpenAiClient, RunnerConfig};
    let client = std::sync::Arc::new(OpenAiClient::new(
        format!("{}/v1", mock_server.uri()),
        "test-key".to_string(),
        "gpt-4o".to_string(),
    ));

    let runner = AgentRunner::new(
        client,
        RunnerConfig {
            concurrency: 2,
            ..Default::default()
        },
    );
    let run_result = runner.run(&parsed.agent, scenarios.clone(), None).await;
    let traces = run_result.traces;

    assert_eq!(traces.len(), scenarios.len());
}

// ─── Gatekeeper tests ──────────────────────────────────────────────────────

#[test]
fn gatekeeper_first_promotion_approved() {
    use agentforge_core::{DimensionScores, Scorecard};
    use agentforge_gatekeeper::{GateStatus, Gatekeeper, GatekeeperConfig};

    let challenger = Scorecard {
        run_id: Uuid::new_v4(),
        agent_id: Uuid::new_v4(),
        agent_name: "test-agent".to_string(),
        agent_version: "1.0.0".to_string(),
        aggregate_score: 0.88,
        pass_rate: 0.88,
        total_scenarios: 100,
        passed: 88,
        failed: 12,
        errors: 0,
        review_needed: 0,
        dimension_scores: DimensionScores {
            task_completion: 0.90,
            tool_selection: 0.85,
            argument_correctness: 0.88,
            schema_compliance: 0.92,
            instruction_adherence: 0.87,
            path_efficiency: 0.80,
        },
        failure_clusters: vec![],
        duration_seconds: 120,
        total_input_tokens: 10000,
        total_output_tokens: 5000,
    };

    let gk = Gatekeeper::new(GatekeeperConfig::default());
    let decision = gk
        .evaluate(
            Uuid::new_v4(),
            Uuid::new_v4(),
            None,
            &challenger,
            &[],
            &[],
            &[0.88, 0.87, 0.89],
        )
        .unwrap();

    assert!(
        decision.approved,
        "First promotion should be approved automatically"
    );
    assert_eq!(decision.gates[0].status, GateStatus::Waived);
    assert_eq!(decision.gates[1].status, GateStatus::Waived);
}

// ─── Optimizer tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn optimizer_generates_variants_without_llm() {
    use agentforge_core::{DimensionScores, Scorecard};
    use agentforge_optimizer::{Optimizer, OptimizerConfig};

    let parsed = parse_agent_file(SAMPLE_AGENT_YAML).unwrap();
    let scorecard = Scorecard {
        run_id: Uuid::new_v4(),
        agent_id: Uuid::new_v4(),
        agent_name: "test-agent".to_string(),
        agent_version: "1.0.0".to_string(),
        aggregate_score: 0.65,
        pass_rate: 0.65,
        total_scenarios: 100,
        passed: 65,
        failed: 35,
        errors: 0,
        review_needed: 0,
        dimension_scores: DimensionScores {
            task_completion: 0.60,
            tool_selection: 0.70,
            argument_correctness: 0.65,
            schema_compliance: 0.72,
            instruction_adherence: 0.58,
            path_efficiency: 0.80,
        },
        failure_clusters: vec![],
        duration_seconds: 60,
        total_input_tokens: 5000,
        total_output_tokens: 2000,
    };

    let optimizer = Optimizer::new(OptimizerConfig {
        min_variants: 2,
        max_variants: 8,
        llm_api_key: "".to_string(), // No LLM — uses deterministic fallbacks
        ..Default::default()
    });

    let result = optimizer
        .generate_variants(&parsed.agent, &scorecard, &[], "sha123")
        .await
        .unwrap();
    assert!(result.variants.len() >= 2);
    assert!(result.variants.len() <= 8);

    // All variants should have a valid system prompt
    for v in &result.variants {
        assert!(!v.agent.system_prompt.is_empty());
        assert_eq!(v.parent_sha, "sha123");
    }
}

// ─── Scorecard diff test ───────────────────────────────────────────────────

#[test]
fn scorecard_diff_computed_correctly() {
    use agentforge_core::DimensionScores;

    let champ_scores = DimensionScores {
        task_completion: 0.80,
        tool_selection: 0.75,
        argument_correctness: 0.78,
        schema_compliance: 0.85,
        instruction_adherence: 0.82,
        path_efficiency: 0.70,
    };

    let challenger_scores = DimensionScores {
        task_completion: 0.87,
        tool_selection: 0.80,
        argument_correctness: 0.83,
        schema_compliance: 0.88,
        instruction_adherence: 0.84,
        path_efficiency: 0.75,
    };

    use agentforge_core::EvalWeights;
    let weights = EvalWeights::default();

    let champ_agg = champ_scores.weighted_aggregate(&weights);
    let challenger_agg = challenger_scores.weighted_aggregate(&weights);

    assert!(challenger_agg > champ_agg, "Challenger should score higher");
    assert!((challenger_agg - champ_agg) > 0.0);
}

// ─── HTTP-level route tests (auth middleware + health + SSE) ───────────────

/// Stub LLM client — never called in these tests; exists only to satisfy the
/// `AppState::llm_client` field type.
struct StubLlmClient;

#[async_trait::async_trait]
impl agentforge_runner::LlmClient for StubLlmClient {
    async fn complete(
        &self,
        _req: agentforge_runner::LlmRequest,
    ) -> agentforge_core::Result<agentforge_runner::LlmResponse> {
        Err(agentforge_core::AgentForgeError::ConfigError(
            "test stub — should not be called".to_string(),
        ))
    }
    fn provider_name(&self) -> &str {
        "stub"
    }
    fn model_id(&self) -> &str {
        "stub-model"
    }
}

/// Build a minimal `AppState` with a lazy (never-connected) Postgres pool.
/// In CI, DATABASE_URL points to the real test postgres (already migrated);
/// locally it falls back to a stub URL — the pool is lazy and only actually
/// connects when a query is executed.
fn make_test_state(api_key: Option<String>) -> std::sync::Arc<agentforge_api::AppState> {
    use std::sync::{atomic::AtomicI64, Arc};
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://stub:stub@localhost/stub_unused_agentforge".to_string());
    let db = sqlx::PgPool::connect_lazy(&db_url).unwrap();
    Arc::new(agentforge_api::AppState {
        db,
        llm_client: Arc::new(StubLlmClient),
        scorer_config: agentforge_scorer::ScorerConfig::default(),
        optimizer_config: agentforge_optimizer::OptimizerConfig::default(),
        gatekeeper_config: agentforge_gatekeeper::GatekeeperConfig::default(),
        trace_exporter: Arc::new(agentforge_observability::NoopExporter),
        active_runs: Arc::new(AtomicI64::new(0)),
        max_concurrent_runs: 10,
        max_scenarios: 2000,
        api_key,
    })
}

/// Convenience: build the full axum router under test.
fn make_test_router(api_key: Option<String>) -> axum::Router {
    agentforge_api::router(make_test_state(api_key))
}

// ── health probe ───────────────────────────────────────────────────────────

#[tokio::test]
async fn health_endpoint_returns_200() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(None);
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

/// The health endpoint must be reachable even when an API key is configured —
/// it is explicitly exempt from authentication.
#[tokio::test]
async fn health_endpoint_exempt_from_auth() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(Some("secret-key".to_string()));
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ── auth: no api_key configured (dev mode) ────────────────────────────────

/// When no API key is configured, requests without Authorization header are allowed.
#[tokio::test]
async fn api_routes_allow_all_without_api_key() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(None);
    // POST /api/v1/agents is a real route — it will hit the DB; expect 500 not 401
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/runs")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    // 500 (DB unreachable) or 200 is fine — what matters is NOT 401
    assert_ne!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "unauthenticated request must not be rejected when no api_key is configured"
    );
}

// ── auth: api_key is set ──────────────────────────────────────────────────

/// A request with no Authorization header must be rejected with 401.
#[tokio::test]
async fn api_routes_reject_missing_auth_header_with_401() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(Some("the-real-key".to_string()));
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/runs")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// A request with a wrong Bearer token must be rejected with 401.
#[tokio::test]
async fn api_routes_reject_wrong_bearer_token_with_401() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(Some("the-real-key".to_string()));
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/runs")
        .header("Authorization", "Bearer wrong-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// A request with the correct Bearer token must pass the auth gate.
/// (The downstream handler may still return 500 because the DB isn't live —
///  we only assert the status is NOT 401.)
#[tokio::test]
async fn api_routes_allow_valid_bearer_token() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(Some("my-valid-key".to_string()));
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/runs")
        .header("Authorization", "Bearer my-valid-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "valid token should pass auth; got {:?}",
        resp.status()
    );
}

/// 401 response must include WWW-Authenticate: Bearer realm="agentforge".
#[tokio::test]
async fn api_routes_401_includes_www_authenticate_header() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(Some("key".to_string()));
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/runs")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let www = resp
        .headers()
        .get("WWW-Authenticate")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        www.contains("Bearer"),
        "Missing 'Bearer' in WWW-Authenticate: {www}"
    );
    assert!(
        www.contains("agentforge"),
        "Missing realm 'agentforge' in WWW-Authenticate: {www}"
    );
}

/// 401 response body must be valid JSON with the expected error envelope.
#[tokio::test]
async fn api_routes_401_body_is_valid_json_error_envelope() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(Some("key".to_string()));
    let req = Request::builder()
        .method("GET")
        .uri("/api/v1/runs")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let bytes = axum::body::to_bytes(resp.into_body(), 8192).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("401 body should be JSON");
    assert_eq!(
        body["error"]["code"], "UNAUTHORIZED",
        "error.code should be UNAUTHORIZED"
    );
    assert!(
        body["error"]["message"].is_string(),
        "error.message should be a string"
    );
}

// ── SSE endpoint: content-type ─────────────────────────────────────────────

/// GET /api/v1/runs/:id/progress must respond with Content-Type: text/event-stream.
/// (The run won't be found — we just verify the SSE framing is set up correctly.)
#[tokio::test]
async fn sse_progress_endpoint_responds_with_event_stream_content_type() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = make_test_router(None);
    let run_id = Uuid::new_v4();
    let req = Request::builder()
        .method("GET")
        .uri(format!("/api/v1/runs/{run_id}/progress"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("text/event-stream"),
        "SSE endpoint should return Content-Type: text/event-stream, got: {ct}"
    );
}

/// The SSE progress endpoint must not require authentication even when an API
/// key is set (matches standard practice of not auth-gating stream endpoints
/// when credentials appear in the URL or via EventSource which can't set headers).
/// Update: The endpoint IS behind auth, so with a key and no header → 401.
#[tokio::test]
async fn sse_progress_endpoint_requires_auth_when_key_is_set() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(Some("my-key".to_string()));
    let run_id = Uuid::new_v4();
    let req = Request::builder()
        .method("GET")
        .uri(format!("/api/v1/runs/{run_id}/progress"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "SSE endpoint must enforce auth just like every other /api/v1/* route"
    );
}

// ── unknown routes ─────────────────────────────────────────────────────────

#[tokio::test]
async fn unknown_route_returns_404() {
    use axum::{body::Body, http::Request, http::StatusCode};
    use tower::ServiceExt;

    let app = make_test_router(None);
    let req = Request::builder()
        .uri("/api/v1/does-not-exist")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ─── Full pipeline test: github-actions-expert.agent.md ───────────────────
//
// This test exercises the complete eval pipeline against a real-world Copilot
// agent file (github/awesome-copilot) without any live DB or paid LLM calls:
//
//   parse → validate → generate scenarios → run (mocked LLM) → score
//
// The fixture lives at fixtures/github-actions-expert.agent.md and is embedded
// at compile time via include_str!.

const GITHUB_ACTIONS_EXPERT_MD: &str =
    include_str!("../../../fixtures/github-actions-expert.agent.md");

// ── 1. Parse ────────────────────────────────────────────────────────────────

#[test]
fn github_actions_expert_parses_as_copilot_agent_md() {
    let parsed = parse_agent_file(GITHUB_ACTIONS_EXPERT_MD).unwrap();
    assert_eq!(parsed.format, AgentFileFormat::CopilotAgentMd);
    assert_eq!(parsed.agent.name, "GitHub Actions Expert");
    // System prompt is the Markdown body — must contain key sections
    assert!(
        parsed.agent.system_prompt.contains("GitHub Actions"),
        "system_prompt should include agent content"
    );
    assert!(
        parsed.agent.system_prompt.contains("security"),
        "system_prompt should include security content"
    );
    // SHA is deterministic for the same content
    assert_eq!(parsed.sha.len(), 64, "SHA should be a hex-encoded SHA-256");
}

#[test]
fn github_actions_expert_tools_parsed() {
    let parsed = parse_agent_file(GITHUB_ACTIONS_EXPERT_MD).unwrap();
    // The frontmatter declares: ['github/*', 'search/codebase', 'edit/editFiles',
    // 'execute/runInTerminal', 'read/readFile', 'search/fileSearch']
    // After namespace stripping → github, codebase, editFiles, runInTerminal, readFile, fileSearch
    assert!(
        !parsed.agent.tools.is_empty(),
        "Expected at least one tool from the frontmatter"
    );
    let tool_names: Vec<&str> = parsed.agent.tools.iter().map(|t| t.name.as_str()).collect();
    assert!(
        tool_names.contains(&"github"),
        "Expected 'github' tool; got: {tool_names:?}"
    );
}

#[test]
fn github_actions_expert_description_in_metadata() {
    let parsed = parse_agent_file(GITHUB_ACTIONS_EXPERT_MD).unwrap();
    if let Some(meta) = parsed.agent.metadata {
        if let Some(desc) = meta["description"].as_str() {
            assert!(
                desc.contains("GitHub Actions"),
                "description should mention GitHub Actions: {desc}"
            );
        }
    }
}

// ── 2. Validate ─────────────────────────────────────────────────────────────

#[test]
fn github_actions_expert_validates_without_hard_errors() {
    let parsed = parse_agent_file(GITHUB_ACTIONS_EXPERT_MD).unwrap();
    let result = validate_agent_file(&parsed.agent);
    // Copilot agent.md files legitimately omit output_schema and constraints;
    // the validator should emit warnings but no hard errors that block a run.
    assert!(
        result.errors.is_empty(),
        "Unexpected validation errors: {:?}",
        result.errors
    );
}

#[test]
fn github_actions_expert_validation_warns_about_missing_schema() {
    let parsed = parse_agent_file(GITHUB_ACTIONS_EXPERT_MD).unwrap();
    let result = validate_agent_file(&parsed.agent);
    // Should have at least one warning (no output_schema / no constraints)
    assert!(
        !result.warnings.is_empty(),
        "Expected at least one validation warning for a schema-free agent.md file"
    );
}

// ── 3. Scenario generation (no LLM) ─────────────────────────────────────────

#[tokio::test]
async fn github_actions_expert_generates_scenarios() {
    let parsed = parse_agent_file(GITHUB_ACTIONS_EXPERT_MD).unwrap();
    let agent_id = Uuid::new_v4();

    let scenarios = generate_scenarios(
        &parsed.agent,
        &ScenarioGeneratorConfig {
            total_count: 10,
            agent_id,
            llm_api_key: None, // deterministic path only
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert!(
        !scenarios.is_empty(),
        "Should generate at least one scenario"
    );
    assert!(scenarios.len() <= 10, "Should not exceed requested count");

    // Every scenario must be associated with the correct agent
    for s in &scenarios {
        assert_eq!(s.agent_id, agent_id);
    }
}

#[tokio::test]
async fn github_actions_expert_adversarial_scenarios() {
    use agentforge_scenarios::adversarial::generate_adversarial_scenarios;
    let parsed = parse_agent_file(GITHUB_ACTIONS_EXPERT_MD).unwrap();
    let scenarios = generate_adversarial_scenarios(&parsed.agent, 8, Uuid::new_v4()).unwrap();
    assert!(
        scenarios.len() >= 3,
        "Expected at least 3 adversarial scenarios; got {}",
        scenarios.len()
    );
    // Adversarial scenarios carry the Adversarial source tag
    use agentforge_core::ScenarioSource;
    assert!(
        scenarios
            .iter()
            .all(|s| s.source == ScenarioSource::Adversarial),
        "All returned scenarios should have Adversarial source"
    );
}

// ── 4. Full pipeline: parse → generate → run (mocked LLM) → score ───────────

#[tokio::test]
async fn github_actions_expert_full_pipeline_with_mocked_llm() {
    use agentforge_runner::{AgentRunner, OpenAiClient, RunnerConfig};
    use agentforge_scorer::{score_run, ScorerConfig};

    let mock_server = MockServer::start().await;

    // Mock LLM: returns a plausible GitHub Actions workflow snippet
    let mock_response = serde_json::json!({
        "id": "chatcmpl-ga-expert",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here is a secure GitHub Actions workflow:\n\n```yaml\nname: CI\non:\n  push:\n    branches: [main]\npermissions:\n  contents: read\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2\n```\n\nKey security notes:\n- Pinned to full commit SHA\n- Minimal permissions (contents: read)\n- OIDC preferred over static credentials"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 120,
            "total_tokens": 320
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&mock_response))
        .mount(&mock_server)
        .await;

    // Parse the real agent file
    let parsed = parse_agent_file(GITHUB_ACTIONS_EXPERT_MD).unwrap();
    let agent_id = Uuid::new_v4();

    // Generate a small scenario set (deterministic — no LLM key)
    let scenarios = generate_scenarios(
        &parsed.agent,
        &ScenarioGeneratorConfig {
            total_count: 5,
            agent_id,
            llm_api_key: None,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert!(!scenarios.is_empty(), "Need at least one scenario to run");

    // Run through mocked LLM
    let client = std::sync::Arc::new(OpenAiClient::new(
        format!("{}/v1", mock_server.uri()),
        "test-key".to_string(),
        "gpt-4o".to_string(),
    ));
    let runner = AgentRunner::new(
        client,
        RunnerConfig {
            concurrency: 2,
            ..Default::default()
        },
    );
    let run_result = runner.run(&parsed.agent, scenarios.clone(), None).await;
    let mut traces = run_result.traces;

    // Every scenario must have a corresponding trace
    assert_eq!(
        traces.len(),
        scenarios.len(),
        "Trace count must match scenario count"
    );

    // Score all traces (no judge key → heuristic scoring only)
    let run_id = Uuid::new_v4();
    let scorer_config = ScorerConfig {
        judge_api_key: String::new(), // no LLM judge
        ..Default::default()
    };
    let scorecard = score_run(
        &mut traces,
        &scenarios,
        &parsed.agent,
        run_id,
        &scorer_config,
    )
    .await
    .unwrap();

    // Basic scorecard sanity checks
    assert_eq!(scorecard.agent_name, "GitHub Actions Expert");
    assert_eq!(scorecard.total_scenarios, scenarios.len() as u32);
    assert!(
        scorecard.aggregate_score >= 0.0 && scorecard.aggregate_score <= 1.0,
        "aggregate_score out of range: {}",
        scorecard.aggregate_score
    );
    // `review_needed` is an orthogonal boolean flag — a Pass-status trace can
    // also have review_needed=true, so the counts are not mutually exclusive.
    // The statuses that are mutually exclusive are Pass/Fail/Error/ReviewNeeded;
    // passed+failed+errors must be ≤ total (ReviewNeeded status makes up the rest).
    assert!(
        scorecard.passed + scorecard.failed + scorecard.errors <= scorecard.total_scenarios,
        "passed+failed+errors exceeds total: {}+{}+{} > {}",
        scorecard.passed,
        scorecard.failed,
        scorecard.errors,
        scorecard.total_scenarios
    );
    assert!(
        scorecard.review_needed <= scorecard.total_scenarios,
        "review_needed count exceeds total scenarios"
    );
    // Mocked LLM always replies → no Error traces expected
    assert_eq!(
        scorecard.errors, 0,
        "No traces should be in Error state when LLM always responds"
    );
}
