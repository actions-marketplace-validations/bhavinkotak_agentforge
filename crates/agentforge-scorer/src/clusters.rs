use agentforge_core::{DimensionScores, FailureCluster, Trace, TraceStatus, TraceStep};

/// Classify the primary failure reason for a trace into one of the known clusters.
pub fn classify_failure_cluster(
    trace: &Trace,
    scores: &DimensionScores,
    failure_reasons: &[String],
) -> FailureCluster {
    if trace.status == TraceStatus::Pass {
        return FailureCluster::NoFailure;
    }

    if trace.status == TraceStatus::Error {
        return FailureCluster::Unknown;
    }

    // --- Hard failures (unambiguous signal, check first) ---

    // Schema violation: output did not conform to required schema
    if scores.schema_compliance < 0.3 {
        return FailureCluster::SchemaViolation;
    }

    // Hallucinated argument: agent made up parameters
    if scores.argument_correctness < 0.3 {
        return FailureCluster::HallucinatedArgument;
    }

    // Looping: many LLM calls with very few tool calls between them
    if detect_loop(trace) {
        return FailureCluster::Looping;
    }

    // No tools called at all despite being needed (pure premature stop)
    if scores.path_efficiency < 0.1 {
        return FailureCluster::PrematureStop;
    }

    // --- Check keyword hints from deterministic failure reasons ---
    let failure_text = failure_reasons.join(" ").to_lowercase();
    if failure_text.contains("wrong_tool") || failure_text.contains("missing required tools") {
        return FailureCluster::WrongTool;
    }
    if failure_text.contains("argument") || failure_text.contains("hallucinated") {
        return FailureCluster::HallucinatedArgument;
    }
    if failure_text.contains("schema") {
        return FailureCluster::SchemaViolation;
    }
    if failure_text.contains("constraint") || failure_text.contains("instruction adherence") {
        return FailureCluster::ConstraintBreach;
    }

    // --- Soft failures: use the weakest dimension to name the primary failure ---
    // This ensures we always return a meaningful cluster rather than Unknown.
    // We compare raw scores; the one furthest from 1.0 is the root cause.
    let candidates = [
        (scores.task_completion, FailureCluster::PrematureStop),
        (scores.tool_selection, FailureCluster::WrongTool),
        (scores.instruction_adherence, FailureCluster::ConstraintBreach),
    ];

    candidates
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, cluster)| cluster.clone())
        .unwrap_or(FailureCluster::Unknown)
}

/// Detect if the agent entered a loop (many repeated LLM calls with no tool calls between them).
fn detect_loop(trace: &Trace) -> bool {
    let llm_count = trace
        .steps
        .iter()
        .filter(|s| matches!(s, TraceStep::LlmCall(_)))
        .count();
    let tool_count = trace
        .steps
        .iter()
        .filter(|s| matches!(s, TraceStep::ToolCall(_)))
        .count();

    // Heuristic: >5 LLM calls with very few tool calls indicates looping
    llm_count > 5 && tool_count <= 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_core::{FailureCluster, TraceStatus};

    fn make_scores(
        tool: f64,
        args: f64,
        schema: f64,
        adherence: f64,
        efficiency: f64,
    ) -> DimensionScores {
        DimensionScores {
            task_completion: 0.5,
            tool_selection: tool,
            argument_correctness: args,
            schema_compliance: schema,
            instruction_adherence: adherence,
            path_efficiency: efficiency,
        }
    }

    fn make_empty_trace(status: TraceStatus) -> Trace {
        Trace {
            id: uuid::Uuid::new_v4(),
            run_id: uuid::Uuid::new_v4(),
            scenario_id: uuid::Uuid::new_v4(),
            status,
            steps: vec![],
            final_output: None,
            scores: None,
            aggregate_score: None,
            failure_cluster: FailureCluster::Unknown,
            failure_reason: None,
            review_needed: false,
            llm_calls: 0,
            tool_invocations: 0,
            input_tokens: 0,
            output_tokens: 0,
            latency_ms: 0,
            retry_count: 0,
            seed: 0,
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn pass_returns_no_failure() {
        let trace = make_empty_trace(TraceStatus::Pass);
        let scores = make_scores(1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::NoFailure
        );
    }

    #[test]
    fn error_returns_unknown() {
        let trace = make_empty_trace(TraceStatus::Error);
        let scores = make_scores(0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::Unknown
        );
    }

    #[test]
    fn low_schema_compliance_is_schema_violation() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = make_scores(1.0, 1.0, 0.1, 1.0, 1.0);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::SchemaViolation
        );
    }

    #[test]
    fn low_tool_selection_is_wrong_tool() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = make_scores(0.1, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::WrongTool
        );
    }

    #[test]
    fn low_args_is_hallucinated_argument() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = make_scores(1.0, 0.1, 1.0, 1.0, 1.0);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::HallucinatedArgument
        );
    }

    #[test]
    fn low_constraint_is_breach() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = make_scores(1.0, 1.0, 1.0, 0.1, 1.0);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::ConstraintBreach
        );
    }

    // ── Regression: Fail trace must never get NoFailure ──────────────────────

    #[test]
    fn fail_trace_with_moderate_scores_never_gets_no_failure() {
        let trace = make_empty_trace(TraceStatus::Fail);
        // Moderate scores (30-70%) - these were the bug: old code returned Unknown
        // but now should return a meaningful cluster via weakest-dimension fallback.
        let scores = DimensionScores {
            task_completion: 0.55,
            tool_selection: 0.65,
            argument_correctness: 0.70,
            schema_compliance: 0.60,
            instruction_adherence: 0.70,
            path_efficiency: 0.75,
        };
        let cluster = classify_failure_cluster(&trace, &scores, &[]);
        assert_ne!(cluster, FailureCluster::NoFailure,
            "Fail trace must never get NoFailure cluster");
    }

    #[test]
    fn review_needed_trace_never_gets_no_failure() {
        let trace = make_empty_trace(TraceStatus::ReviewNeeded);
        let scores = DimensionScores {
            task_completion: 0.5,
            tool_selection: 0.9,
            argument_correctness: 0.9,
            schema_compliance: 0.9,
            instruction_adherence: 0.9,
            path_efficiency: 0.9,
        };
        let cluster = classify_failure_cluster(&trace, &scores, &[]);
        assert_ne!(cluster, FailureCluster::NoFailure,
            "ReviewNeeded trace must never get NoFailure cluster");
    }

    // ── Weakest-dimension fallback tests ─────────────────────────────────────

    #[test]
    fn weakest_task_completion_yields_premature_stop() {
        let trace = make_empty_trace(TraceStatus::Fail);
        // task_completion is the weakest (0.3 < 0.6 < 0.7)
        let scores = DimensionScores {
            task_completion: 0.3,
            tool_selection: 0.6,
            argument_correctness: 0.7,
            schema_compliance: 0.7,
            instruction_adherence: 0.6,
            path_efficiency: 0.5,
        };
        let cluster = classify_failure_cluster(&trace, &scores, &[]);
        assert_eq!(cluster, FailureCluster::PrematureStop,
            "Weakest task_completion should yield PrematureStop");
    }

    #[test]
    fn weakest_tool_selection_yields_wrong_tool() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = DimensionScores {
            task_completion: 0.6,
            tool_selection: 0.3,
            argument_correctness: 0.7,
            schema_compliance: 0.7,
            instruction_adherence: 0.6,
            path_efficiency: 0.5,
        };
        let cluster = classify_failure_cluster(&trace, &scores, &[]);
        assert_eq!(cluster, FailureCluster::WrongTool,
            "Weakest tool_selection should yield WrongTool");
    }

    #[test]
    fn weakest_instruction_adherence_yields_constraint_breach() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = DimensionScores {
            task_completion: 0.6,
            tool_selection: 0.6,
            argument_correctness: 0.7,
            schema_compliance: 0.7,
            instruction_adherence: 0.2,
            path_efficiency: 0.6,
        };
        let cluster = classify_failure_cluster(&trace, &scores, &[]);
        assert_eq!(cluster, FailureCluster::ConstraintBreach,
            "Weakest instruction_adherence should yield ConstraintBreach");
    }

    // ── Hard-failure priority tests ───────────────────────────────────────────

    #[test]
    fn schema_violation_beats_moderate_task_completion() {
        let trace = make_empty_trace(TraceStatus::Fail);
        // schema is below hard threshold even though task is also low
        let scores = DimensionScores {
            task_completion: 0.2,
            tool_selection: 0.9,
            argument_correctness: 0.9,
            schema_compliance: 0.2,  // < 0.3
            instruction_adherence: 0.9,
            path_efficiency: 0.9,
        };
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::SchemaViolation,
            "Schema violation should take priority over weak task_completion"
        );
    }

    #[test]
    fn hallucinated_arg_beats_weak_dimensions() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = DimensionScores {
            task_completion: 0.3,
            tool_selection: 0.3,
            argument_correctness: 0.1,  // < 0.3 hard threshold
            schema_compliance: 0.9,
            instruction_adherence: 0.3,
            path_efficiency: 0.3,
        };
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::HallucinatedArgument
        );
    }

    // ── Keyword hint tests ────────────────────────────────────────────────────

    #[test]
    fn wrong_tool_keyword_triggers_cluster() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = make_scores(0.7, 0.9, 0.9, 0.9, 0.7);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &["wrong_tool".to_string()]),
            FailureCluster::WrongTool
        );
    }

    #[test]
    fn missing_required_tools_keyword_triggers_wrong_tool() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = make_scores(0.7, 0.9, 0.9, 0.9, 0.7);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &["missing required tools".to_string()]),
            FailureCluster::WrongTool
        );
    }

    #[test]
    fn argument_keyword_triggers_hallucinated_argument() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = make_scores(0.7, 0.9, 0.9, 0.9, 0.7);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &["argument mismatch".to_string()]),
            FailureCluster::HallucinatedArgument
        );
    }

    #[test]
    fn constraint_keyword_triggers_constraint_breach() {
        let trace = make_empty_trace(TraceStatus::Fail);
        let scores = make_scores(0.7, 0.9, 0.9, 0.9, 0.7);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &["instruction adherence failed".to_string()]),
            FailureCluster::ConstraintBreach
        );
    }

    // ── Loop detection ────────────────────────────────────────────────────────

    #[test]
    fn many_llm_calls_few_tools_triggers_looping() {
        let mut trace = make_empty_trace(TraceStatus::Fail);
        use agentforge_core::{LlmCallStep, TraceStep};
        use chrono::Utc;
        for i in 0..6 {
            trace.steps.push(TraceStep::LlmCall(LlmCallStep {
                index: i,
                model: "gpt-4o".to_string(),
                messages: vec![],
                response: serde_json::json!({}),
                input_tokens: 50,
                output_tokens: 20,
                latency_ms: 500,
                timestamp: Utc::now(),
            }));
        }
        // Only one tool call — satisfies loop heuristic (>5 LLM, <=1 tool)
        use agentforge_core::{ToolCallStep, TraceStep as TS};
        trace.steps.push(TS::ToolCall(ToolCallStep {
            index: 6,
            tool_name: "search".to_string(),
            call_id: "c1".to_string(),
            arguments: serde_json::json!({}),
            timestamp: Utc::now(),
        }));
        let scores = make_scores(0.5, 0.5, 0.9, 0.9, 0.2);
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::Looping
        );
    }

    #[test]
    fn few_llm_calls_does_not_trigger_looping() {
        let mut trace = make_empty_trace(TraceStatus::Fail);
        use agentforge_core::{LlmCallStep, TraceStep};
        use chrono::Utc;
        for i in 0..3 {
            trace.steps.push(TraceStep::LlmCall(LlmCallStep {
                index: i,
                model: "gpt-4o".to_string(),
                messages: vec![],
                response: serde_json::json!({}),
                input_tokens: 50,
                output_tokens: 20,
                latency_ms: 500,
                timestamp: Utc::now(),
            }));
        }
        let scores = make_scores(0.5, 0.5, 0.9, 0.9, 0.2);
        assert_ne!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::Looping
        );
    }

    // ── Path efficiency hard threshold ────────────────────────────────────────

    #[test]
    fn very_low_path_efficiency_is_premature_stop() {
        let trace = make_empty_trace(TraceStatus::Fail);
        // path_efficiency < 0.1 hard threshold
        let scores = DimensionScores {
            task_completion: 0.7,
            tool_selection: 0.7,
            argument_correctness: 0.9,
            schema_compliance: 0.9,
            instruction_adherence: 0.7,
            path_efficiency: 0.05,
        };
        assert_eq!(
            classify_failure_cluster(&trace, &scores, &[]),
            FailureCluster::PrematureStop
        );
    }
}
