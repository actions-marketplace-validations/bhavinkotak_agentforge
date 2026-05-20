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
    // Return a generic success result shaped to match common tool outputs
    serde_json::json!({
        "tool": tool_name,
        "status": "success",
        "result": {
            "message": format!("Tool '{}' executed successfully", tool_name),
            "args_received": args
        }
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
}
