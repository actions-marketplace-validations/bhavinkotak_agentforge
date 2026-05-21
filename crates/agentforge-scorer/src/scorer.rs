use crate::{
    clusters::classify_failure_cluster,
    deterministic::run_deterministic_checks,
    judge::{heuristic_task_completion, run_llm_judge},
};
use agentforge_core::{
    AgentFile, DimensionScores, EvalWeights, FailureCluster, Result, Scenario, Scorecard, Trace,
    TraceStatus,
};
use uuid::Uuid;

/// Configuration for the trace scorer.
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Judge model ID — MUST differ from agent model.
    pub judge_model: String,
    /// Judge LLM base URL (OpenAI-compatible)
    pub judge_base_url: String,
    pub judge_api_key: String,
    /// Confidence threshold below which a trace is flagged for human review.
    pub review_confidence_threshold: f64,
    pub weights: EvalWeights,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            judge_model: "gpt-4o".to_string(),
            judge_base_url: "https://api.openai.com/v1".to_string(),
            judge_api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            review_confidence_threshold: 0.5,
            weights: EvalWeights::default(),
        }
    }
}

/// The main trace scorer.
pub struct TraceScorer {
    #[allow(dead_code)]
    config: ScorerConfig,
}

impl TraceScorer {
    pub fn new(config: ScorerConfig) -> Self {
        Self { config }
    }
}

/// Score a single trace and update it with scores and status.
pub async fn score_trace(
    trace: &mut Trace,
    scenario: &Scenario,
    agent: &AgentFile,
    config: &ScorerConfig,
) -> Result<()> {
    if trace.status == TraceStatus::Error {
        // Error traces get 0.0 aggregate score, no need to score further
        trace.aggregate_score = Some(0.0);
        return Ok(());
    }

    // 1. Run deterministic checks (cheap, no LLM required)
    let det = run_deterministic_checks(trace, scenario, agent);

    // 2. Run LLM judge for semantic dimensions
    let (task_completion, instruction_adherence) = if !config.judge_api_key.is_empty() {
        match run_llm_judge(trace, scenario, agent, config).await {
            Ok(judge_result) => (
                judge_result.task_completion,
                judge_result.instruction_adherence,
            ),
            Err(e) => {
                tracing::warn!(error = %e, "LLM judge failed, using heuristic fallback");
                let tc = heuristic_task_completion(trace, scenario);
                (tc, det.instruction_adherence.clone())
            }
        }
    } else {
        // No judge configured — use heuristic
        let tc = heuristic_task_completion(trace, scenario);
        (tc, det.instruction_adherence.clone())
    };

    let scores = DimensionScores {
        task_completion: task_completion.value,
        tool_selection: det.tool_selection.value,
        argument_correctness: det.argument_correctness.value,
        schema_compliance: det.schema_compliance.value,
        instruction_adherence: instruction_adherence.value,
        path_efficiency: det.path_efficiency.value,
    };

    let aggregate = scores.weighted_aggregate(&config.weights);

    // 3. Check if human review is needed (low confidence scores)
    let min_confidence = [
        task_completion.confidence,
        det.tool_selection.confidence,
        det.argument_correctness.confidence,
        det.schema_compliance.confidence,
        instruction_adherence.confidence,
        det.path_efficiency.confidence,
    ]
    .iter()
    .cloned()
    .fold(f64::MAX, f64::min);

    let review_needed = min_confidence < config.review_confidence_threshold;

    // 4. Determine trace status first (cluster classifier reads trace.status)
    let status = if aggregate >= 0.85 {
        TraceStatus::Pass
    } else if review_needed {
        TraceStatus::ReviewNeeded
    } else {
        TraceStatus::Fail
    };

    trace.scores = Some(scores.clone());
    trace.aggregate_score = Some(aggregate);
    trace.status = status;
    trace.review_needed = review_needed;

    // 5. Classify failure cluster — must come after trace.status is finalised
    //    so classify_failure_cluster sees the correct (Pass/Fail/ReviewNeeded)
    //    status rather than the runner's initial Pass placeholder.
    let failure_cluster = classify_failure_cluster(trace, &scores, &det.failure_reasons);
    trace.failure_cluster = failure_cluster.clone();

    if !det.failure_reasons.is_empty() {
        trace.failure_reason = Some(det.failure_reasons.join("; "));
    }

    Ok(())
}

/// Score a full batch of traces and build the run scorecard.
pub async fn score_run(
    traces: &mut [Trace],
    scenarios: &[Scenario],
    agent: &AgentFile,
    run_id: Uuid,
    config: &ScorerConfig,
) -> Result<Scorecard> {
    let scenario_map: std::collections::HashMap<Uuid, &Scenario> =
        scenarios.iter().map(|s| (s.id, s)).collect();

    for trace in traces.iter_mut() {
        if let Some(scenario) = scenario_map.get(&trace.scenario_id) {
            if let Err(e) = score_trace(trace, scenario, agent, config).await {
                tracing::error!(
                    trace_id = %trace.id,
                    error = %e,
                    "Failed to score trace"
                );
            }
        }
    }

    // Aggregate scores
    let total = traces.len() as u32;
    let passed = traces
        .iter()
        .filter(|t| t.status == TraceStatus::Pass)
        .count() as u32;
    let failed = traces
        .iter()
        .filter(|t| t.status == TraceStatus::Fail)
        .count() as u32;
    let errors = traces
        .iter()
        .filter(|t| t.status == TraceStatus::Error)
        .count() as u32;
    let review = traces.iter().filter(|t| t.review_needed).count() as u32;

    let pass_rate = if total > 0 {
        passed as f64 / total as f64
    } else {
        0.0
    };

    let avg_scores = average_dimension_scores(traces);
    let aggregate_score = avg_scores.weighted_aggregate(&config.weights);

    let failure_clusters = build_failure_cluster_summary(traces);

    let (total_input_tokens, total_output_tokens) =
        traces.iter().fold((0u64, 0u64), |(i, o), t| {
            (i + t.input_tokens as u64, o + t.output_tokens as u64)
        });

    let duration_seconds = traces.iter().map(|t| t.latency_ms).sum::<u64>() / 1000;

    Ok(Scorecard {
        run_id,
        agent_id: traces.first().map(|t| t.run_id).unwrap_or(Uuid::nil()),
        agent_name: agent.name.clone(),
        agent_version: agent.version.clone(),
        aggregate_score,
        pass_rate,
        total_scenarios: total,
        passed,
        failed,
        errors,
        review_needed: review,
        dimension_scores: avg_scores,
        failure_clusters,
        duration_seconds,
        total_input_tokens,
        total_output_tokens,
    })
}

fn average_dimension_scores(traces: &[Trace]) -> DimensionScores {
    let scorable: Vec<&DimensionScores> = traces.iter().filter_map(|t| t.scores.as_ref()).collect();

    if scorable.is_empty() {
        return DimensionScores::default();
    }

    let n = scorable.len() as f64;
    DimensionScores {
        task_completion: scorable.iter().map(|s| s.task_completion).sum::<f64>() / n,
        tool_selection: scorable.iter().map(|s| s.tool_selection).sum::<f64>() / n,
        argument_correctness: scorable.iter().map(|s| s.argument_correctness).sum::<f64>() / n,
        schema_compliance: scorable.iter().map(|s| s.schema_compliance).sum::<f64>() / n,
        instruction_adherence: scorable
            .iter()
            .map(|s| s.instruction_adherence)
            .sum::<f64>()
            / n,
        path_efficiency: scorable.iter().map(|s| s.path_efficiency).sum::<f64>() / n,
    }
}

fn build_failure_cluster_summary(traces: &[Trace]) -> Vec<agentforge_core::FailureClusterSummary> {
    use std::collections::HashMap;
    let mut cluster_counts: HashMap<FailureCluster, (u32, Vec<Uuid>)> = HashMap::new();

    let failed_traces: Vec<&Trace> = traces
        .iter()
        .filter(|t| t.status == TraceStatus::Fail || t.status == TraceStatus::Error)
        .collect();

    for trace in &failed_traces {
        let entry = cluster_counts
            .entry(trace.failure_cluster.clone())
            .or_default();
        entry.0 += 1;
        if entry.1.len() < 3 {
            entry.1.push(trace.scenario_id);
        }
    }

    // Percentage is relative to ALL scenarios (not just failed) so the UI
    // shows "X% of runs hit this failure mode" rather than a within-failures
    // fraction that sums to 100% (confusing when only 1 cluster exists).
    let total = traces.len().max(1) as f64;
    cluster_counts
        .into_iter()
        .map(
            |(cluster, (count, samples))| agentforge_core::FailureClusterSummary {
                percentage: count as f64 / total,
                cluster,
                count,
                sample_scenarios: samples,
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_core::{
        DifficultyTier, FinalOutputStep, ModelConfig, ModelProvider, ScenarioExpected,
        ScenarioInput, ScenarioSource, TraceStep,
    };
    use chrono::Utc;

    fn make_passing_trace(run_id: Uuid, scenario_id: Uuid) -> Trace {
        Trace {
            id: Uuid::new_v4(),
            run_id,
            scenario_id,
            status: TraceStatus::Pass,
            steps: vec![TraceStep::FinalOutput(FinalOutputStep {
                index: 0,
                output: serde_json::json!({"response": "Here is the information you requested about your order."}),
                timestamp: Utc::now(),
            })],
            final_output: Some(
                serde_json::json!({"response": "Here is the information you requested about your order."}),
            ),
            scores: None,
            aggregate_score: None,
            failure_cluster: FailureCluster::NoFailure,
            failure_reason: None,
            review_needed: false,
            llm_calls: 1,
            tool_invocations: 0,
            input_tokens: 50,
            output_tokens: 30,
            latency_ms: 800,
            retry_count: 0,
            seed: 0,
            created_at: Utc::now(),
        }
    }

    fn make_simple_scenario() -> Scenario {
        Scenario {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            input: ScenarioInput {
                user_message: "What is the status of my order?".to_string(),
                conversation_history: vec![],
                context: None,
            },
            expected: ScenarioExpected {
                tool_calls: vec![],
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"response": {"type": "string"}},
                    "required": ["response"]
                })),
                pass_criteria: "Agent should provide a helpful response about the order."
                    .to_string(),
                min_turns: None,
                max_turns: None,
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
                temperature: None,
                max_tokens: None,
                top_p: None,
            },
            system_prompt: "You are helpful.".to_string(),
            tools: vec![],
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"]
            })),
            constraints: vec![],
            eval_hints: None,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn score_trace_no_judge() {
        let agent = make_simple_agent();
        let scenario = make_simple_scenario();
        let run_id = Uuid::new_v4();
        let mut trace = make_passing_trace(run_id, scenario.id);

        // Config with no judge API key
        let config = ScorerConfig {
            judge_api_key: "".to_string(),
            judge_model: "gpt-4o-judge".to_string(), // different from agent model
            ..Default::default()
        };

        score_trace(&mut trace, &scenario, &agent, &config)
            .await
            .unwrap();
        assert!(trace.aggregate_score.is_some());
        assert!(trace.scores.is_some());
    }

    #[tokio::test]
    async fn score_trace_error_status_gets_zero() {
        let agent = make_simple_agent();
        let scenario = make_simple_scenario();
        let run_id = Uuid::new_v4();
        let mut trace = make_passing_trace(run_id, scenario.id);
        trace.status = TraceStatus::Error;

        let config = ScorerConfig {
            judge_api_key: "".to_string(),
            judge_model: "gpt-4o-judge".to_string(),
            ..Default::default()
        };

        score_trace(&mut trace, &scenario, &agent, &config)
            .await
            .unwrap();
        assert_eq!(trace.aggregate_score, Some(0.0));
    }

    #[test]
    fn average_scores_correct() {
        let run_id = Uuid::new_v4();
        let mut traces = vec![
            make_passing_trace(run_id, Uuid::new_v4()),
            make_passing_trace(run_id, Uuid::new_v4()),
        ];
        traces[0].scores = Some(DimensionScores {
            task_completion: 1.0,
            tool_selection: 1.0,
            argument_correctness: 1.0,
            schema_compliance: 1.0,
            instruction_adherence: 1.0,
            path_efficiency: 1.0,
        });
        traces[1].scores = Some(DimensionScores {
            task_completion: 0.5,
            tool_selection: 0.5,
            argument_correctness: 0.5,
            schema_compliance: 0.5,
            instruction_adherence: 0.5,
            path_efficiency: 0.5,
        });

        let avg = average_dimension_scores(&traces);
        assert!((avg.task_completion - 0.75).abs() < 1e-9);
    }

    /// Regression test: failure_cluster must be set AFTER trace.status is finalised.
    /// Previously classify_failure_cluster ran before trace.status was updated so
    /// it always saw TraceStatus::Pass (the runner's initial value) and returned
    /// FailureCluster::NoFailure for every scored trace, including Fail ones.
    #[tokio::test]
    async fn failure_cluster_is_not_no_failure_for_fail_trace() {
        let agent = make_simple_agent();
        let scenario = make_simple_scenario();
        let run_id = Uuid::new_v4();

        // A trace that starts as Pass (runner default) but has very low scores
        // so the scorer will mark it Fail.
        let mut trace = make_passing_trace(run_id, scenario.id);
        trace.tool_invocations = 0; // no tools called — low path efficiency
        trace.llm_calls = 1;

        let config = ScorerConfig {
            judge_api_key: "".to_string(),
            judge_model: "gpt-4o-judge".to_string(),
            ..Default::default()
        };

        score_trace(&mut trace, &scenario, &agent, &config)
            .await
            .unwrap();

        // The trace must NOT carry NoFailure when it ends up as Fail or ReviewNeeded
        if trace.status == TraceStatus::Fail || trace.status == TraceStatus::ReviewNeeded {
            assert_ne!(
                trace.failure_cluster,
                FailureCluster::NoFailure,
                "Fail/ReviewNeeded trace must not have NoFailure cluster; got {trace:?}"
            );
        }
    }

    /// Regression test: build_failure_cluster_summary percentage is relative to
    /// ALL traces (not just failed ones) so it reads as "X% of all scenarios".
    #[test]
    fn cluster_percentage_is_fraction_of_all_scenarios() {
        let run_id = Uuid::new_v4();
        let mut traces = vec![
            make_passing_trace(run_id, Uuid::new_v4()),
            make_passing_trace(run_id, Uuid::new_v4()),
            make_passing_trace(run_id, Uuid::new_v4()),
            make_passing_trace(run_id, Uuid::new_v4()),
            make_passing_trace(run_id, Uuid::new_v4()),
        ];
        // Make 2 of them Fail with WrongTool cluster
        traces[0].status = TraceStatus::Fail;
        traces[0].failure_cluster = FailureCluster::WrongTool;
        traces[1].status = TraceStatus::Fail;
        traces[1].failure_cluster = FailureCluster::WrongTool;

        let summary = build_failure_cluster_summary(&traces);
        let wrong_tool = summary
            .iter()
            .find(|s| s.cluster == FailureCluster::WrongTool)
            .expect("WrongTool cluster should be present");

        assert_eq!(wrong_tool.count, 2);
        // 2 out of 5 total = 40%, NOT 100% of failed
        assert!((wrong_tool.percentage - 0.4).abs() < 1e-9,
            "Expected 0.4 (40% of all scenarios), got {}", wrong_tool.percentage);
    }
}
