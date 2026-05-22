use crate::llm::{LlmClient, LlmMessage, LlmRequest, LlmRole};
use agentforge_core::{
    AgentFile, AgentForgeError, FailureCluster, FinalOutputStep, LlmCallStep, Result, Scenario,
    ToolCallStep, ToolResultStep, Trace, TraceStatus, TraceStep,
};
use chrono::Utc;
use futures::future::join_all;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use uuid::Uuid;

/// Configuration for the agent runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub concurrency: usize,
    pub max_retries: u32,
    pub retry_base_delay_ms: u64,
    pub max_turns: u32,
    pub run_id: Uuid,
    pub seed: u32,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            concurrency: 10,
            max_retries: 3,
            retry_base_delay_ms: 1000,
            max_turns: 20,
            run_id: Uuid::new_v4(),
            seed: 0,
        }
    }
}

/// Result of a full run across all scenarios.
#[derive(Debug)]
pub struct RunResult {
    pub traces: Vec<Trace>,
    pub total_duration: Duration,
}

/// The agent runner orchestrates parallel execution of scenarios.
pub struct AgentRunner {
    llm: Arc<dyn LlmClient>,
    config: RunnerConfig,
}

impl AgentRunner {
    pub fn new(llm: Arc<dyn LlmClient>, config: RunnerConfig) -> Self {
        Self { llm, config }
    }

    /// Execute all scenarios in parallel (up to `concurrency` workers).
    pub async fn run(
        &self,
        agent: &AgentFile,
        scenarios: Vec<Scenario>,
        on_progress: Option<Arc<dyn Fn(u32, u32) + Send + Sync>>,
    ) -> RunResult {
        let semaphore = Arc::new(Semaphore::new(self.config.concurrency));
        let total = scenarios.len() as u32;
        let completed = Arc::new(std::sync::atomic::AtomicU32::new(0));

        let start = Instant::now();

        let futures: Vec<_> = scenarios
            .into_iter()
            .map(|scenario| {
                let sem = semaphore.clone();
                let llm = self.llm.clone();
                let agent = agent.clone();
                let config = self.config.clone();
                let completed = completed.clone();
                let on_progress = on_progress.clone();

                tokio::spawn(async move {
                    let _permit = sem.acquire().await.expect("semaphore not closed");
                    let trace = run_single_with_retry(&llm, &agent, &scenario, &config).await;
                    let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if let Some(cb) = &on_progress {
                        cb(done, total);
                    }
                    trace
                })
            })
            .collect();

        let results = join_all(futures).await;
        let traces = results
            .into_iter()
            .filter_map(|r| r.ok()) // ignore tokio join errors
            .collect();

        RunResult {
            traces,
            total_duration: start.elapsed(),
        }
    }
}

/// Run a single scenario with retry logic.
async fn run_single_with_retry(
    llm: &Arc<dyn LlmClient>,
    agent: &AgentFile,
    scenario: &Scenario,
    config: &RunnerConfig,
) -> Trace {
    let mut retry_count = 0;

    loop {
        match run_single(llm, agent, scenario, config, retry_count).await {
            Ok(mut trace) => {
                trace.retry_count = retry_count;
                return trace;
            }
            Err(e) => {
                let is_transient = matches!(
                    &e,
                    AgentForgeError::RateLimitExceeded { .. }
                        | AgentForgeError::HttpError(_)
                        | AgentForgeError::Timeout { .. }
                );

                if is_transient && retry_count < config.max_retries {
                    retry_count += 1;
                    let delay = config.retry_base_delay_ms * (1 << retry_count.min(5));
                    tracing::warn!(
                        scenario_id = %scenario.id,
                        retry = retry_count,
                        delay_ms = delay,
                        error = %e,
                        "Transient error, retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                } else {
                    // Persistent failure — log so the error is visible in CI
                    tracing::error!(
                        scenario_id = %scenario.id,
                        scenario_input = %scenario.input.user_message,
                        retry_count = retry_count,
                        error = %e,
                        "Scenario failed (non-transient error)"
                    );
                    return error_trace(scenario, config, retry_count, &e);
                }
            }
        }
    }
}

/// Execute a single scenario against the agent.
async fn run_single(
    llm: &Arc<dyn LlmClient>,
    agent: &AgentFile,
    scenario: &Scenario,
    config: &RunnerConfig,
    retry_count: u32,
) -> Result<Trace> {
    let start = Instant::now();
    let mut steps: Vec<TraceStep> = Vec::new();
    let mut step_index = 0u32;
    let mut total_input_tokens = 0u32;
    let mut total_output_tokens = 0u32;
    let mut tool_invocations = 0u32;
    let mut llm_calls = 0u32;
    let mut final_output: Option<serde_json::Value> = None;

    // Build the initial message list
    let mut messages: Vec<LlmMessage> = vec![LlmMessage {
        role: LlmRole::System,
        content: Some(agent.system_prompt.clone()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];

    // Add conversation history
    for turn in &scenario.input.conversation_history {
        messages.push(LlmMessage {
            role: match turn.role {
                agentforge_core::ConversationRole::User => LlmRole::User,
                agentforge_core::ConversationRole::Assistant => LlmRole::Assistant,
                agentforge_core::ConversationRole::System => LlmRole::System,
                agentforge_core::ConversationRole::Tool => LlmRole::Tool,
            },
            content: Some(turn.content.clone()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    // Add the user message for this scenario
    messages.push(LlmMessage {
        role: LlmRole::User,
        content: Some(scenario.input.user_message.clone()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    // Build tool definitions in OpenAI format
    let tools: Option<Vec<serde_json::Value>> = if agent.tools.is_empty() {
        None
    } else {
        Some(
            agent
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect(),
        )
    };

    // Agentic loop
    for _turn in 0..config.max_turns {
        let request = LlmRequest {
            model: agent.model.model_id.clone(),
            messages: messages.clone(),
            tools: tools.clone(),
            temperature: agent.model.temperature,
            max_tokens: agent.model.max_tokens,
            top_p: agent.model.top_p,
        };

        tracing::debug!(
            scenario_id = %scenario.id,
            turn = _turn,
            model = %request.model,
            num_messages = request.messages.len(),
            has_tools = request.tools.is_some(),
            provider = %llm.provider_name(),
            "Sending LLM request"
        );
        let llm_call_start = Instant::now();
        let response = match llm.complete(request.clone()).await {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    scenario_id = %scenario.id,
                    turn = _turn,
                    model = %request.model,
                    provider = %llm.provider_name(),
                    error = %e,
                    "LLM call failed"
                );
                return Err(e);
            }
        };
        let llm_latency = llm_call_start.elapsed().as_millis() as u64;
        llm_calls += 1;
        tracing::debug!(
            scenario_id = %scenario.id,
            turn = _turn,
            model = %response.model,
            input_tokens = response.input_tokens,
            output_tokens = response.output_tokens,
            finish_reason = %response.finish_reason,
            has_tool_calls = response.message.tool_calls.is_some(),
            latency_ms = llm_latency,
            "LLM response received"
        );
        total_input_tokens += response.input_tokens;
        total_output_tokens += response.output_tokens;

        // Record the LLM call step
        steps.push(TraceStep::LlmCall(LlmCallStep {
            index: step_index,
            model: response.model.clone(),
            messages: messages
                .iter()
                .map(|m| serde_json::to_value(m).unwrap_or_default())
                .collect(),
            response: response.raw_response.clone(),
            input_tokens: response.input_tokens,
            output_tokens: response.output_tokens,
            latency_ms: llm_latency,
            timestamp: Utc::now(),
        }));
        step_index += 1;

        // Handle tool calls (only when the parsed list is non-empty;
        // Some([]) is already normalised to None in the LLM parser but
        // guard here too to avoid leaving an assistant msg as the last entry).
        if let Some(tool_calls) = response
            .message
            .tool_calls
            .as_ref()
            .filter(|v| !v.is_empty())
        {
            // Add assistant message with tool calls to history
            messages.push(LlmMessage {
                role: LlmRole::Assistant,
                content: response.message.content.clone(),
                tool_calls: Some(tool_calls.clone()),
                tool_call_id: None,
                name: None,
            });

            for tc in tool_calls {
                let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| serde_json::json!({}));

                steps.push(TraceStep::ToolCall(ToolCallStep {
                    index: step_index,
                    tool_name: tc.function.name.clone(),
                    call_id: tc.id.clone(),
                    arguments: args.clone(),
                    timestamp: Utc::now(),
                }));
                step_index += 1;
                tool_invocations += 1;

                // Simulate tool execution (return a structured mock result)
                let tool_result = simulate_tool_result(&tc.function.name, &args);
                steps.push(TraceStep::ToolResult(ToolResultStep {
                    index: step_index,
                    tool_name: tc.function.name.clone(),
                    call_id: tc.id.clone(),
                    result: tool_result.clone(),
                    is_error: false,
                    timestamp: Utc::now(),
                }));
                step_index += 1;

                // Add tool result to messages.
                // Per OpenAI spec: role=tool, content=<result string>, tool_call_id=<id>.
                // Do NOT include `name` — it is not part of the tool-result message spec
                // and confuses vLLM-backed endpoints (e.g. NVIDIA NIM Mistral).
                messages.push(LlmMessage {
                    role: LlmRole::Tool,
                    content: Some(serde_json::to_string(&tool_result).unwrap_or_default()),
                    tool_calls: None,
                    tool_call_id: Some(tc.id.clone()),
                    name: None,
                });
            }
            // Continue the loop for the next LLM response
        } else {
            // No tool calls — final response
            let output_text = response.message.content.clone().unwrap_or_default();
            let output = serde_json::json!({ "response": output_text });

            steps.push(TraceStep::FinalOutput(FinalOutputStep {
                index: step_index,
                output: output.clone(),
                timestamp: Utc::now(),
            }));
            final_output = Some(output);
            break;
        }
    }

    let latency_ms = start.elapsed().as_millis() as u64;

    Ok(Trace {
        id: Uuid::new_v4(),
        run_id: config.run_id,
        scenario_id: scenario.id,
        status: TraceStatus::Pass, // Will be scored later
        steps,
        final_output,
        scores: None,
        aggregate_score: None,
        failure_cluster: FailureCluster::NoFailure,
        failure_reason: None,
        review_needed: false,
        llm_calls,
        tool_invocations,
        input_tokens: total_input_tokens,
        output_tokens: total_output_tokens,
        latency_ms,
        retry_count,
        seed: config.seed,
        created_at: Utc::now(),
    })
}

/// Create an error trace when a scenario cannot be executed.
fn error_trace(
    scenario: &Scenario,
    config: &RunnerConfig,
    retry_count: u32,
    error: &AgentForgeError,
) -> Trace {
    Trace {
        id: Uuid::new_v4(),
        run_id: config.run_id,
        scenario_id: scenario.id,
        status: TraceStatus::Error,
        steps: vec![],
        final_output: None,
        scores: None,
        aggregate_score: None,
        failure_cluster: FailureCluster::Unknown,
        failure_reason: Some(error.to_string()),
        review_needed: false,
        llm_calls: 0,
        tool_invocations: 0,
        input_tokens: 0,
        output_tokens: 0,
        latency_ms: 0,
        retry_count,
        seed: config.seed,
        created_at: Utc::now(),
    }
}

/// Simulate tool execution — returns a plausible result.
/// In production, tools would be real API calls or stubs provided by the user.
fn simulate_tool_result(tool_name: &str, args: &serde_json::Value) -> serde_json::Value {
    let query = args
        .get("query")
        .or_else(|| args.get("input"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let file_path = args
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or(".github/workflows/ci.yml");
    let command = args
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let name_lower = tool_name.to_lowercase();

    // GitHub search/API
    if name_lower.contains("github") {
        return serde_json::json!({
            "results": [
                {
                    "path": ".github/workflows/ci.yml",
                    "repository": "owner/repo",
                    "content": "name: CI\non:\n  push:\n    branches: [main]\n  pull_request:\n    branches: [main]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-node@v4\n        with:\n          node-version: '20'\n      - run: npm ci\n      - run: npm test",
                    "url": "https://github.com/owner/repo/blob/main/.github/workflows/ci.yml"
                },
                {
                    "path": ".github/workflows/deploy.yml",
                    "repository": "owner/repo",
                    "content": "name: Deploy\non:\n  push:\n    branches: [main]\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - run: npm ci && npm run build\n      - uses: peaceiris/actions-gh-pages@v3\n        with:\n          github_token: ${{ secrets.GITHUB_TOKEN }}\n          publish_dir: ./dist",
                    "url": "https://github.com/owner/repo/blob/main/.github/workflows/deploy.yml"
                }
            ],
            "total_count": 2,
            "query": query
        });
    }

    // File search
    if name_lower.contains("filesearch") || (name_lower.contains("search") && name_lower.contains("file")) {
        return serde_json::json!({
            "files": [
                ".github/workflows/ci.yml",
                ".github/workflows/deploy.yml",
                ".github/workflows/release.yml",
                "package.json",
                "README.md"
            ],
            "query": query
        });
    }

    // Codebase / semantic search
    if name_lower.contains("codebase") || name_lower.contains("searchcodebase") {
        return serde_json::json!({
            "matches": [
                {
                    "file": ".github/workflows/ci.yml",
                    "snippet": "on:\n  push:\n    branches: [main]\n  pull_request:",
                    "line": 2
                },
                {
                    "file": "package.json",
                    "snippet": "\"scripts\": {\n  \"test\": \"jest\",\n  \"build\": \"tsc\"\n}",
                    "line": 5
                }
            ],
            "query": query
        });
    }

    // Read file
    if name_lower.contains("readfile") || name_lower.contains("read") {
        let content = if file_path.contains("ci.yml") || file_path.contains("workflow") {
            "name: CI\non:\n  push:\n    branches: [main]\n  pull_request:\n    branches: [main]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-node@v4\n        with:\n          node-version: '20'\n          cache: 'npm'\n      - run: npm ci\n      - run: npm run lint\n      - run: npm test\n      - run: npm run build"
        } else if file_path.contains("package.json") {
            "{\n  \"name\": \"my-app\",\n  \"version\": \"1.0.0\",\n  \"scripts\": {\n    \"build\": \"tsc\",\n    \"test\": \"jest --coverage\",\n    \"lint\": \"eslint src/\"\n  },\n  \"engines\": { \"node\": \">=18\" }\n}"
        } else {
            "# My Project\n\nThis project uses GitHub Actions for CI/CD.\n\n## Workflows\n- `ci.yml` — runs tests on every PR\n- `deploy.yml` — deploys to production on merge to main"
        };
        return serde_json::json!({
            "path": file_path,
            "content": content,
            "exists": true
        });
    }

    // Edit files
    if name_lower.contains("editfiles") || name_lower.contains("edit") {
        return serde_json::json!({
            "status": "success",
            "path": file_path,
            "message": format!("File '{}' updated successfully", file_path),
            "lines_changed": 12
        });
    }

    // Run in terminal
    if name_lower.contains("terminal") || name_lower.contains("run") {
        let output = if command.contains("npm") || command.contains("yarn") {
            "added 847 packages in 12s\n✓ All tests passed (24 suites, 156 tests)"
        } else if command.contains("git") {
            "On branch main\nYour branch is up to date with 'origin/main'.\nnothing to commit, working tree clean"
        } else if command.contains("gh") || command.contains("act") {
            "✓ Run .github/workflows/ci.yml\n  ✓ build (ubuntu-latest)\n    ✓ Checkout code\n    ✓ Set up Node.js 20\n    ✓ npm ci\n    ✓ npm test\nAll jobs passed."
        } else {
            "Command completed successfully with exit code 0."
        };
        return serde_json::json!({
            "stdout": output,
            "stderr": "",
            "exit_code": 0,
            "command": command
        });
    }

    // Generic fallback — still provide structured content
    serde_json::json!({
        "tool": tool_name,
        "status": "success",
        "result": format!("Executed '{}' with args: {}", tool_name, args),
        "args_received": args
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmMessage, LlmResponse, LlmRole};
    use agentforge_core::{
        DifficultyTier, ModelConfig, ModelProvider, ScenarioExpected, ScenarioInput, ScenarioSource,
    };
    use async_trait::async_trait;
    use mockall::mock;
    use mockall::predicate::*;
    use std::sync::Arc;

    mock! {
        TestLlm {}

        #[async_trait]
        impl LlmClient for TestLlm {
            async fn complete(&self, request: LlmRequest) -> Result<LlmResponse>;
            fn provider_name(&self) -> &str;
            fn model_id(&self) -> &str;
        }
    }

    fn make_scenario(agent_id: Uuid) -> Scenario {
        Scenario {
            id: Uuid::new_v4(),
            agent_id,
            input: ScenarioInput {
                user_message: "Hello, can you help me?".to_string(),
                conversation_history: vec![],
                context: None,
            },
            expected: ScenarioExpected {
                tool_calls: vec![],
                output_schema: None,
                pass_criteria: "Agent should greet the user.".to_string(),
                min_turns: Some(1),
                max_turns: Some(5),
            },
            difficulty: DifficultyTier::Easy,
            domain: None,
            source: ScenarioSource::SchemaDerived,
            tags: vec![],
            created_at: Utc::now(),
        }
    }

    fn make_simple_agent() -> AgentFile {
        AgentFile {
            agentforge_schema_version: "1".to_string(),
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model: ModelConfig {
                provider: ModelProvider::Openai,
                model_id: "gpt-4o".to_string(),
                temperature: Some(0.2),
                max_tokens: Some(1024),
                top_p: None,
            },
            system_prompt: "You are a helpful assistant.".to_string(),
            tools: vec![],
            output_schema: None,
            constraints: vec![],
            eval_hints: None,
            metadata: None,
        }
    }

    fn make_final_response() -> LlmResponse {
        LlmResponse {
            model: "gpt-4o".to_string(),
            message: LlmMessage {
                role: LlmRole::Assistant,
                content: Some("Hello! I'm here to help you.".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: "stop".to_string(),
            input_tokens: 50,
            output_tokens: 20,
            latency_ms: 500,
            raw_response: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn runner_produces_trace_on_success() {
        let mut mock_llm = MockTestLlm::new();
        mock_llm
            .expect_complete()
            .times(1)
            .returning(|_| Ok(make_final_response()));
        mock_llm
            .expect_provider_name()
            .return_const("openai".to_string());
        mock_llm
            .expect_model_id()
            .return_const("gpt-4o".to_string());

        let agent = make_simple_agent();
        let agent_id = Uuid::new_v4();
        let scenario = make_scenario(agent_id);
        let config = RunnerConfig::default();
        let runner = AgentRunner::new(Arc::new(mock_llm), config);

        let result = runner.run(&agent, vec![scenario], None).await;
        assert_eq!(result.traces.len(), 1);
        assert_ne!(result.traces[0].status, TraceStatus::Error);
        assert!(result.traces[0].final_output.is_some());
    }

    #[tokio::test]
    async fn runner_marks_error_on_persistent_failure() {
        let mut mock_llm = MockTestLlm::new();
        mock_llm
            .expect_complete()
            .times(..) // any number of retries
            .returning(|_| {
                Err(AgentForgeError::LlmError {
                    provider: "openai".to_string(),
                    message: "Persistent error".to_string(),
                })
            });
        mock_llm
            .expect_provider_name()
            .return_const("openai".to_string());
        mock_llm
            .expect_model_id()
            .return_const("gpt-4o".to_string());

        let agent = make_simple_agent();
        let agent_id = Uuid::new_v4();
        let scenario = make_scenario(agent_id);
        let config = RunnerConfig {
            max_retries: 0, // no retries for this test
            ..Default::default()
        };
        let runner = AgentRunner::new(Arc::new(mock_llm), config);

        let result = runner.run(&agent, vec![scenario], None).await;
        assert_eq!(result.traces.len(), 1);
        assert_eq!(result.traces[0].status, TraceStatus::Error);
        assert!(result.traces[0].failure_reason.is_some());
    }

    #[tokio::test]
    async fn runner_runs_concurrently() {
        let mut mock_llm = MockTestLlm::new();
        mock_llm
            .expect_complete()
            .times(5)
            .returning(|_| Ok(make_final_response()));
        mock_llm
            .expect_provider_name()
            .return_const("openai".to_string());
        mock_llm
            .expect_model_id()
            .return_const("gpt-4o".to_string());

        let agent = make_simple_agent();
        let agent_id = Uuid::new_v4();
        let scenarios: Vec<_> = (0..5).map(|_| make_scenario(agent_id)).collect();
        let config = RunnerConfig {
            concurrency: 5,
            ..Default::default()
        };
        let runner = AgentRunner::new(Arc::new(mock_llm), config);
        let result = runner.run(&agent, scenarios, None).await;
        assert_eq!(result.traces.len(), 5);
    }

    #[test]
    fn simulate_tool_result_returns_success() {
        let result = simulate_tool_result("get_order", &serde_json::json!({"order_id": "ORD-123"}));
        assert_eq!(result["status"].as_str(), Some("success"));
    }

    // ── simulate_tool_result: GitHub tool ────────────────────────────────────

    #[test]
    fn simulate_github_returns_results_array() {
        let result = simulate_tool_result("github_search", &serde_json::json!({"query": "ci.yml"}));
        assert!(result["results"].is_array(), "github_search must return results array");
    }

    #[test]
    fn simulate_github_results_are_non_empty() {
        let result = simulate_tool_result("github_search", &serde_json::json!({"query": "workflows"}));
        let arr = result["results"].as_array().unwrap();
        assert!(!arr.is_empty(), "results must not be empty");
    }

    #[test]
    fn simulate_github_result_has_path_field() {
        let result = simulate_tool_result("github", &serde_json::json!({"query": "test"}));
        let first = &result["results"][0];
        assert!(first["path"].is_string(), "each result must have a path");
    }

    #[test]
    fn simulate_github_result_content_contains_checkout_action() {
        let result = simulate_tool_result("github_search", &serde_json::json!({}));
        let content = result["results"][0]["content"].as_str().unwrap_or("");
        assert!(content.contains("actions/checkout"), "GitHub mock must reference actions/checkout");
    }

    #[test]
    fn simulate_github_returns_total_count() {
        let result = simulate_tool_result("github", &serde_json::json!({}));
        assert!(result["total_count"].as_u64().is_some(), "must have total_count");
    }

    #[test]
    fn simulate_github_echoes_query_arg() {
        let result = simulate_tool_result("github_search", &serde_json::json!({"query": "my-special-query"}));
        let q = result["query"].as_str().unwrap_or("");
        assert_eq!(q, "my-special-query", "query should be echoed back");
    }

    #[test]
    fn simulate_github_uppercase_name_still_works() {
        let result = simulate_tool_result("GITHUB", &serde_json::json!({}));
        assert!(result["results"].is_array());
    }

    // ── simulate_tool_result: fileSearch ────────────────────────────────────

    #[test]
    fn simulate_filesearch_returns_files_array() {
        let result = simulate_tool_result("fileSearch", &serde_json::json!({"query": "yml"}));
        assert!(result["files"].is_array(), "fileSearch must return files array");
    }

    #[test]
    fn simulate_filesearch_files_are_non_empty() {
        let result = simulate_tool_result("fileSearch", &serde_json::json!({}));
        let arr = result["files"].as_array().unwrap();
        assert!(!arr.is_empty());
    }

    #[test]
    fn simulate_filesearch_has_workflow_path() {
        let result = simulate_tool_result("fileSearch", &serde_json::json!({}));
        let files: Vec<&str> = result["files"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        let has_workflow = files.iter().any(|f| f.contains(".github/workflows"));
        assert!(has_workflow, "fileSearch results should include workflow files");
    }

    #[test]
    fn simulate_file_search_snake_case_name() {
        let result = simulate_tool_result("file_search", &serde_json::json!({}));
        assert!(result["files"].is_array());
    }

    // ── simulate_tool_result: readFile ───────────────────────────────────────

    #[test]
    fn simulate_readfile_returns_content_field() {
        let result = simulate_tool_result("readFile", &serde_json::json!({"file_path": "ci.yml"}));
        assert!(result["content"].is_string(), "readFile must return content field");
    }

    #[test]
    fn simulate_readfile_content_is_non_empty() {
        let result = simulate_tool_result("readFile", &serde_json::json!({"file_path": "ci.yml"}));
        let content = result["content"].as_str().unwrap_or("");
        assert!(!content.is_empty(), "readFile content must not be empty");
    }

    #[test]
    fn simulate_readfile_has_path_field() {
        let result = simulate_tool_result("readFile", &serde_json::json!({"file_path": "readme.md"}));
        assert!(result["path"].is_string());
    }

    #[test]
    fn simulate_readfile_ci_yml_has_checkout() {
        let result = simulate_tool_result("readFile", &serde_json::json!({"file_path": ".github/workflows/ci.yml"}));
        let content = result["content"].as_str().unwrap_or("");
        assert!(content.contains("actions/checkout"), "ci.yml mock should reference actions/checkout");
    }

    #[test]
    fn simulate_readfile_package_json_has_test_script() {
        let result = simulate_tool_result("readFile", &serde_json::json!({"file_path": "package.json"}));
        let content = result["content"].as_str().unwrap_or("");
        assert!(content.contains("test"), "package.json mock should reference test script");
    }

    #[test]
    fn simulate_readfile_exists_is_true() {
        let result = simulate_tool_result("readFile", &serde_json::json!({}));
        assert_eq!(result["exists"].as_bool(), Some(true));
    }

    // ── simulate_tool_result: editFiles ──────────────────────────────────────

    #[test]
    fn simulate_editfiles_returns_success_status() {
        let result = simulate_tool_result("editFiles", &serde_json::json!({"file_path": "x.yml"}));
        assert_eq!(result["status"].as_str(), Some("success"));
    }

    #[test]
    fn simulate_editfiles_has_lines_changed() {
        let result = simulate_tool_result("editFiles", &serde_json::json!({}));
        assert!(result["lines_changed"].as_u64().is_some(), "editFiles must have lines_changed");
    }

    #[test]
    fn simulate_editfiles_lines_changed_nonzero() {
        let result = simulate_tool_result("editFiles", &serde_json::json!({}));
        assert!(result["lines_changed"].as_u64().unwrap_or(0) > 0);
    }

    #[test]
    fn simulate_editfiles_has_path_field() {
        let result = simulate_tool_result("editFiles", &serde_json::json!({"file_path": "src/main.rs"}));
        assert!(result["path"].is_string());
    }

    // ── simulate_tool_result: runInTerminal ──────────────────────────────────

    #[test]
    fn simulate_terminal_returns_stdout() {
        let result = simulate_tool_result("runInTerminal", &serde_json::json!({"command": "npm test"}));
        assert!(result["stdout"].is_string(), "terminal must have stdout field");
    }

    #[test]
    fn simulate_terminal_stdout_is_non_empty() {
        let result = simulate_tool_result("runInTerminal", &serde_json::json!({"command": "npm ci"}));
        let stdout = result["stdout"].as_str().unwrap_or("");
        assert!(!stdout.is_empty());
    }

    #[test]
    fn simulate_terminal_exit_code_zero() {
        let result = simulate_tool_result("runInTerminal", &serde_json::json!({"command": "ls"}));
        assert_eq!(result["exit_code"].as_i64(), Some(0));
    }

    #[test]
    fn simulate_terminal_npm_returns_test_output() {
        let result = simulate_tool_result("runInTerminal", &serde_json::json!({"command": "npm test"}));
        let stdout = result["stdout"].as_str().unwrap_or("");
        assert!(stdout.contains("test") || stdout.contains("pass"), "npm test output should mention tests");
    }

    #[test]
    fn simulate_terminal_git_returns_branch_info() {
        let result = simulate_tool_result("runInTerminal", &serde_json::json!({"command": "git status"}));
        let stdout = result["stdout"].as_str().unwrap_or("");
        assert!(stdout.contains("branch") || stdout.contains("main"), "git status should mention branch");
    }

    #[test]
    fn simulate_terminal_gh_act_returns_workflow_result() {
        let result = simulate_tool_result("runInTerminal", &serde_json::json!({"command": "gh act"}));
        let stdout = result["stdout"].as_str().unwrap_or("");
        assert!(stdout.contains("job") || stdout.contains("workflow") || stdout.contains("passed"),
            "gh act output should mention workflow result");
    }

    #[test]
    fn simulate_terminal_has_stderr_field() {
        let result = simulate_tool_result("runInTerminal", &serde_json::json!({}));
        assert!(result["stderr"].is_string());
    }

    // ── simulate_tool_result: codebase search ────────────────────────────────

    #[test]
    fn simulate_codebase_returns_matches() {
        let result = simulate_tool_result("codebase", &serde_json::json!({"query": "push"}));
        assert!(result["matches"].is_array(), "codebase must return matches array");
    }

    #[test]
    fn simulate_codebase_matches_non_empty() {
        let result = simulate_tool_result("codebase", &serde_json::json!({}));
        assert!(!result["matches"].as_array().unwrap().is_empty());
    }

    #[test]
    fn simulate_codebase_match_has_file_field() {
        let result = simulate_tool_result("codebase", &serde_json::json!({}));
        let first = &result["matches"][0];
        assert!(first["file"].is_string());
    }

    #[test]
    fn simulate_searchcodebase_also_works() {
        let result = simulate_tool_result("searchCodebase", &serde_json::json!({}));
        assert!(result["matches"].is_array());
    }

    // ── simulate_tool_result: fallback / unknown tools ───────────────────────

    #[test]
    fn simulate_unknown_tool_returns_valid_json() {
        let result = simulate_tool_result("totally_unknown_tool", &serde_json::json!({}));
        // Should be a JSON object, not null or error
        assert!(result.is_object());
    }

    #[test]
    fn simulate_unknown_tool_has_status_success() {
        let result = simulate_tool_result("mystery_tool_xyz", &serde_json::json!({}));
        assert_eq!(result["status"].as_str(), Some("success"));
    }

    #[test]
    fn simulate_unknown_tool_includes_tool_name() {
        let result = simulate_tool_result("my_custom_tool", &serde_json::json!({}));
        let tool_field = result["tool"].as_str().unwrap_or("");
        assert_eq!(tool_field, "my_custom_tool");
    }

    // ── runner mechanics ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn runner_increments_tool_invocations() {
        use crate::llm::{ToolCall, ToolCallFunction};
        let tool_response = LlmResponse {
            model: "gpt-4o".to_string(),
            message: LlmMessage {
                role: LlmRole::Assistant,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "github_search".to_string(),
                        arguments: r#"{"query":"ci"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            finish_reason: "tool_calls".to_string(),
            input_tokens: 50,
            output_tokens: 10,
            latency_ms: 200,
            raw_response: serde_json::json!({}),
        };

        let mut mock_llm = MockTestLlm::new();
        // First call returns a tool call, second returns final answer
        mock_llm
            .expect_complete()
            .times(1)
            .returning(move |_| Ok(tool_response.clone()));
        mock_llm
            .expect_complete()
            .times(1)
            .returning(|_| Ok(make_final_response()));
        mock_llm
            .expect_provider_name()
            .return_const("openai".to_string());
        mock_llm
            .expect_model_id()
            .return_const("gpt-4o".to_string());

        let agent = make_simple_agent();
        let scenario = make_scenario(Uuid::new_v4());
        let config = RunnerConfig::default();
        let runner = AgentRunner::new(Arc::new(mock_llm), config);
        let result = runner.run(&agent, vec![scenario], None).await;

        assert_eq!(result.traces.len(), 1);
        assert_eq!(result.traces[0].tool_invocations, 1);
        assert_eq!(result.traces[0].llm_calls, 2);
    }

    #[tokio::test]
    async fn runner_records_final_output() {
        let mut mock_llm = MockTestLlm::new();
        mock_llm
            .expect_complete()
            .times(1)
            .returning(|_| Ok(make_final_response()));
        mock_llm
            .expect_provider_name()
            .return_const("openai".to_string());
        mock_llm
            .expect_model_id()
            .return_const("gpt-4o".to_string());

        let agent = make_simple_agent();
        let runner = AgentRunner::new(
            Arc::new(mock_llm),
            RunnerConfig::default(),
        );
        let result = runner.run(&agent, vec![make_scenario(Uuid::new_v4())], None).await;
        assert!(result.traces[0].final_output.is_some());
        let text = result.traces[0].final_output.as_ref().unwrap()["response"]
            .as_str()
            .unwrap_or("");
        assert!(!text.is_empty());
    }

    #[tokio::test]
    async fn runner_stops_at_max_turns() {
        use crate::llm::{ToolCall, ToolCallFunction};
        // Always return a tool call so the agent loops
        let mut mock_llm = MockTestLlm::new();
        mock_llm
            .expect_complete()
            .times(..)
            .returning(|_| {
                Ok(LlmResponse {
                    model: "gpt-4o".to_string(),
                    message: LlmMessage {
                        role: LlmRole::Assistant,
                        content: None,
                        tool_calls: Some(vec![ToolCall {
                            id: "call_x".to_string(),
                            tool_type: "function".to_string(),
                            function: ToolCallFunction {
                                name: "github_search".to_string(),
                                arguments: r#"{}"#.to_string(),
                            },
                        }]),
                        tool_call_id: None,
                        name: None,
                    },
                    finish_reason: "tool_calls".to_string(),
                    input_tokens: 10,
                    output_tokens: 5,
                    latency_ms: 50,
                    raw_response: serde_json::json!({}),
                })
            });
        mock_llm
            .expect_provider_name()
            .return_const("openai".to_string());
        mock_llm
            .expect_model_id()
            .return_const("gpt-4o".to_string());

        let agent = make_simple_agent();
        let config = RunnerConfig {
            max_turns: 3,
            max_retries: 0,
            ..Default::default()
        };
        let runner = AgentRunner::new(Arc::new(mock_llm), config);
        let result = runner.run(&agent, vec![make_scenario(Uuid::new_v4())], None).await;

        assert_eq!(result.traces.len(), 1);
        // Agent ran exactly max_turns (3) LLM calls
        assert_eq!(result.traces[0].llm_calls, 3);
    }

    #[tokio::test]
    async fn runner_reports_progress_callback() {
        let mut mock_llm = MockTestLlm::new();
        mock_llm
            .expect_complete()
            .times(3)
            .returning(|_| Ok(make_final_response()));
        mock_llm
            .expect_provider_name()
            .return_const("openai".to_string());
        mock_llm
            .expect_model_id()
            .return_const("gpt-4o".to_string());

        let agent_id = Uuid::new_v4();
        let scenarios: Vec<_> = (0..3).map(|_| make_scenario(agent_id)).collect();
        let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter2 = counter.clone();

        let on_progress: Arc<dyn Fn(u32, u32) + Send + Sync> = Arc::new(move |done, _total| {
            counter2.store(done, std::sync::atomic::Ordering::SeqCst);
        });

        let agent = make_simple_agent();
        let runner = AgentRunner::new(Arc::new(mock_llm), RunnerConfig::default());
        let result = runner.run(&agent, scenarios, Some(on_progress)).await;
        assert_eq!(result.traces.len(), 3);
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn runner_error_trace_has_failure_reason() {
        let mut mock_llm = MockTestLlm::new();
        mock_llm
            .expect_complete()
            .times(..)
            .returning(|_| {
                Err(AgentForgeError::LlmError {
                    provider: "openai".to_string(),
                    message: "timeout".to_string(),
                })
            });
        mock_llm
            .expect_provider_name()
            .return_const("openai".to_string());
        mock_llm
            .expect_model_id()
            .return_const("gpt-4o".to_string());

        let agent = make_simple_agent();
        let runner = AgentRunner::new(
            Arc::new(mock_llm),
            RunnerConfig { max_retries: 0, ..Default::default() },
        );
        let result = runner.run(&agent, vec![make_scenario(Uuid::new_v4())], None).await;
        assert_eq!(result.traces[0].status, TraceStatus::Error);
        assert!(result.traces[0].failure_reason.is_some());
        let reason = result.traces[0].failure_reason.as_ref().unwrap();
        assert!(!reason.is_empty());
    }
}
