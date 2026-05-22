use crate::mutations::{
    inject_few_shot_examples, rewrite_prompt, rewrite_tool_descriptions, tighten_output_schema,
};
use agentforge_core::{AgentFile, Result, Scorecard, Trace};

/// Configuration for the optimizer.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub min_variants: usize,
    pub max_variants: usize,
    /// OpenAI-compatible base URL for LLM-powered mutations
    pub llm_base_url: String,
    pub llm_api_key: String,
    pub llm_model: String,
    /// Minimum number of passing traces before attempting few-shot injection
    pub few_shot_min_traces: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            min_variants: 5,
            max_variants: 20,
            llm_base_url: "https://api.openai.com/v1".to_string(),
            llm_api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            llm_model: "gpt-4o".to_string(),
            few_shot_min_traces: 50,
        }
    }
}

/// A candidate agent variant with its mutation description.
#[derive(Debug, Clone)]
pub struct AgentVariant {
    pub agent: AgentFile,
    pub mutation_type: MutationType,
    pub description: String,
    pub parent_sha: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationType {
    PromptRewrite,
    ToolDescriptionRewrite,
    OutputSchemaTighten,
    FewShotInjection,
    InstructionReorder,
    ModelDowngrade,
}

impl std::fmt::Display for MutationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MutationType::PromptRewrite => write!(f, "prompt_rewrite"),
            MutationType::ToolDescriptionRewrite => write!(f, "tool_description_rewrite"),
            MutationType::OutputSchemaTighten => write!(f, "output_schema_tighten"),
            MutationType::FewShotInjection => write!(f, "few_shot_injection"),
            MutationType::InstructionReorder => write!(f, "instruction_reorder"),
            MutationType::ModelDowngrade => write!(f, "model_downgrade"),
        }
    }
}

/// The result of an optimization cycle.
#[derive(Debug)]
pub struct OptimizationResult {
    pub variants: Vec<AgentVariant>,
    pub mutation_types_applied: Vec<MutationType>,
}

/// The optimizer generates candidate variants of an agent.
pub struct Optimizer {
    config: OptimizerConfig,
}

impl Optimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Generate candidate variants based on the current scorecard and failure analysis.
    pub async fn generate_variants(
        &self,
        agent: &AgentFile,
        scorecard: &Scorecard,
        passing_traces: &[Trace],
        parent_sha: &str,
    ) -> Result<OptimizationResult> {
        let mut variants = Vec::new();
        let mut mutation_types = Vec::new();

        tracing::info!(
            agent = %agent.name,
            aggregate_score = scorecard.aggregate_score,
            "Starting optimization cycle"
        );

        // Priority 1: Prompt rewrite (if task completion or instruction adherence is low)
        if scorecard.dimension_scores.task_completion < 0.8
            || scorecard.dimension_scores.instruction_adherence < 0.8
        {
            let n = (self.config.max_variants / 5).max(2);
            match rewrite_prompt(agent, scorecard, &self.config, n).await {
                Ok(mut prompt_variants) => {
                    for pv in prompt_variants.drain(..) {
                        variants.push(AgentVariant {
                            agent: pv,
                            mutation_type: MutationType::PromptRewrite,
                            description: "Prompt rewritten for clarity and constraint tightening"
                                .to_string(),
                            parent_sha: parent_sha.to_string(),
                        });
                    }
                    mutation_types.push(MutationType::PromptRewrite);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Prompt rewrite mutation failed");
                    // Apply a deterministic fallback
                    let fallback = reorder_instructions(agent, parent_sha);
                    variants.push(fallback);
                    mutation_types.push(MutationType::InstructionReorder);
                }
            }
        }

        // Priority 2: Tool description rewrite (if tool selection or arg correctness is low)
        if !agent.tools.is_empty()
            && (scorecard.dimension_scores.tool_selection < 0.8
                || scorecard.dimension_scores.argument_correctness < 0.8)
        {
            let n = (self.config.max_variants / 5).max(2);
            match rewrite_tool_descriptions(agent, scorecard, &self.config, n).await {
                Ok(mut tool_variants) => {
                    for tv in tool_variants.drain(..) {
                        variants.push(AgentVariant {
                            agent: tv,
                            mutation_type: MutationType::ToolDescriptionRewrite,
                            description:
                                "Tool descriptions rewritten with examples and type constraints"
                                    .to_string(),
                            parent_sha: parent_sha.to_string(),
                        });
                    }
                    mutation_types.push(MutationType::ToolDescriptionRewrite);
                }
                Err(e) => tracing::warn!(error = %e, "Tool description rewrite failed"),
            }
        }

        // Priority 3: Output schema tightening (if schema compliance is low)
        if scorecard.dimension_scores.schema_compliance < 0.85 {
            if let Some(schema_variant) = tighten_output_schema(agent) {
                variants.push(AgentVariant {
                    agent: schema_variant,
                    mutation_type: MutationType::OutputSchemaTighten,
                    description: "Output schema tightened with stricter required fields"
                        .to_string(),
                    parent_sha: parent_sha.to_string(),
                });
                mutation_types.push(MutationType::OutputSchemaTighten);
            }
        }

        // Priority 4: Few-shot example injection (if enough passing traces)
        if passing_traces.len() >= self.config.few_shot_min_traces {
            match inject_few_shot_examples(agent, passing_traces) {
                Ok(few_shot_variant) => {
                    variants.push(AgentVariant {
                        agent: few_shot_variant,
                        mutation_type: MutationType::FewShotInjection,
                        description: format!(
                            "Injected {} few-shot examples from top-scoring traces",
                            passing_traces.len().min(5)
                        ),
                        parent_sha: parent_sha.to_string(),
                    });
                    mutation_types.push(MutationType::FewShotInjection);
                }
                Err(e) => tracing::warn!(error = %e, "Few-shot injection failed"),
            }
        }

        // Ensure we have at least min_variants
        while variants.len() < self.config.min_variants {
            variants.push(reorder_instructions(agent, parent_sha));
        }

        // Truncate to max_variants
        variants.truncate(self.config.max_variants);

        tracing::info!(
            variants = variants.len(),
            mutations = ?mutation_types.iter().map(|m| m.to_string()).collect::<Vec<_>>(),
            "Optimization cycle complete"
        );

        Ok(OptimizationResult {
            variants,
            mutation_types_applied: mutation_types,
        })
    }
}

/// Deterministic fallback: reorder instructions to put critical ones first.
fn reorder_instructions(agent: &AgentFile, parent_sha: &str) -> AgentVariant {
    let mut new_agent = agent.clone();

    // Sort constraints by priority (put "never" constraints first, then "always")
    let mut constraints = new_agent.constraints.clone();
    constraints.sort_by(|a, b| {
        let a_never = a.to_lowercase().starts_with("never");
        let b_never = b.to_lowercase().starts_with("never");
        b_never.cmp(&a_never)
    });
    new_agent.constraints = constraints;

    // Prepend a summary of critical constraints to the system prompt
    if !new_agent.constraints.is_empty() {
        let critical = new_agent
            .constraints
            .iter()
            .take(3)
            .map(|c| format!("- {}", c))
            .collect::<Vec<_>>()
            .join("\n");

        let new_prompt = format!(
            "CRITICAL RULES (always follow these first):\n{}\n\n{}",
            critical, new_agent.system_prompt
        );
        new_agent.system_prompt = new_prompt;
    }

    AgentVariant {
        agent: new_agent,
        mutation_type: MutationType::InstructionReorder,
        description: "Critical instructions moved to the top of the system prompt".to_string(),
        parent_sha: parent_sha.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_core::{DimensionScores, ModelConfig, ModelProvider, Scorecard};
    use uuid::Uuid;

    fn make_agent() -> AgentFile {
        AgentFile {
            agentforge_schema_version: "1".to_string(),
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model: ModelConfig {
                provider: ModelProvider::Openai,
                model_id: "gpt-4o".to_string(),
                temperature: Some(0.2),
                max_tokens: Some(2048),
                top_p: None,
            },
            system_prompt: "You are a helpful assistant.\nAlways be polite.".to_string(),
            tools: vec![],
            output_schema: None,
            constraints: vec![
                "Never share personal data.".to_string(),
                "Always confirm before deleting.".to_string(),
            ],
            eval_hints: None,
            metadata: None,
        }
    }

    fn make_scorecard(agg: f64, tc: f64, ts: f64, ac: f64, sc: f64, ia: f64, pe: f64) -> Scorecard {
        Scorecard {
            run_id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            agent_name: "test-agent".to_string(),
            agent_version: "1.0.0".to_string(),
            aggregate_score: agg,
            pass_rate: 0.7,
            total_scenarios: 10,
            passed: 7,
            failed: 3,
            errors: 0,
            review_needed: 0,
            dimension_scores: DimensionScores {
                task_completion: tc,
                tool_selection: ts,
                argument_correctness: ac,
                schema_compliance: sc,
                instruction_adherence: ia,
                path_efficiency: pe,
            },
            failure_clusters: vec![],
            duration_seconds: 60,
            total_input_tokens: 5000,
            total_output_tokens: 2000,
        }
    }

    #[test]
    fn reorder_puts_never_constraints_first() {
        let agent = make_agent();
        let variant = reorder_instructions(&agent, "sha_test");
        assert!(variant.agent.system_prompt.starts_with("CRITICAL RULES"));
        assert!(variant.agent.system_prompt.contains("Never share"));
    }

    // ── reorder_instructions ─────────────────────────────────────────────────

    #[test]
    fn reorder_mutation_type_is_instruction_reorder() {
        let agent = make_agent();
        let variant = reorder_instructions(&agent, "abc");
        assert_eq!(variant.mutation_type, MutationType::InstructionReorder);
    }

    #[test]
    fn reorder_parent_sha_is_propagated() {
        let agent = make_agent();
        let sha = "test_sha_123";
        let variant = reorder_instructions(&agent, sha);
        assert_eq!(variant.parent_sha, sha);
    }

    #[test]
    fn reorder_never_constraint_sorted_before_always() {
        let mut agent = make_agent();
        agent.constraints = vec![
            "Always be polite.".to_string(),
            "Never share PII.".to_string(),
            "Always confirm before deleting.".to_string(),
        ];
        let variant = reorder_instructions(&agent, "sha");
        // "Never share PII." should be first in constraints
        assert!(
            variant.agent.constraints[0]
                .to_lowercase()
                .starts_with("never"),
            "Never constraints must come first after reorder"
        );
    }

    #[test]
    fn reorder_no_crash_with_empty_constraints() {
        let mut agent = make_agent();
        agent.constraints = vec![];
        // Should not panic
        let variant = reorder_instructions(&agent, "sha");
        // Without constraints, prompt should remain unchanged (no "CRITICAL RULES" prefix)
        assert!(
            !variant.agent.system_prompt.starts_with("CRITICAL RULES"),
            "With no constraints, system_prompt should not get CRITICAL RULES prefix"
        );
    }

    #[test]
    fn reorder_preserves_original_system_prompt_content() {
        let agent = make_agent();
        let original = agent.system_prompt.clone();
        let variant = reorder_instructions(&agent, "sha");
        assert!(
            variant.agent.system_prompt.contains(&original),
            "Original system prompt must be preserved in reordered variant"
        );
    }

    #[test]
    fn reorder_description_is_non_empty() {
        let agent = make_agent();
        let variant = reorder_instructions(&agent, "sha");
        assert!(!variant.description.is_empty());
    }

    // ── MutationType Display ──────────────────────────────────────────────────

    #[test]
    fn mutation_type_display_prompt_rewrite() {
        assert_eq!(MutationType::PromptRewrite.to_string(), "prompt_rewrite");
    }

    #[test]
    fn mutation_type_display_tool_description_rewrite() {
        assert_eq!(
            MutationType::ToolDescriptionRewrite.to_string(),
            "tool_description_rewrite"
        );
    }

    #[test]
    fn mutation_type_display_output_schema_tighten() {
        assert_eq!(
            MutationType::OutputSchemaTighten.to_string(),
            "output_schema_tighten"
        );
    }

    #[test]
    fn mutation_type_display_few_shot_injection() {
        assert_eq!(
            MutationType::FewShotInjection.to_string(),
            "few_shot_injection"
        );
    }

    #[test]
    fn mutation_type_display_instruction_reorder() {
        assert_eq!(
            MutationType::InstructionReorder.to_string(),
            "instruction_reorder"
        );
    }

    #[test]
    fn mutation_type_display_model_downgrade() {
        assert_eq!(MutationType::ModelDowngrade.to_string(), "model_downgrade");
    }

    // ── generate_variants: deterministic (no LLM) ────────────────────────────

    #[tokio::test]
    async fn generates_at_least_min_variants() {
        let agent = make_agent();
        let scorecard = make_scorecard(0.6, 0.5, 0.7, 0.7, 0.5, 0.5, 0.7);
        let config = OptimizerConfig {
            min_variants: 3,
            max_variants: 10,
            llm_api_key: "".to_string(), // No LLM
            llm_model: "gpt-4o".to_string(),
            ..Default::default()
        };
        let optimizer = Optimizer::new(config);
        let result = optimizer
            .generate_variants(&agent, &scorecard, &[], "abc123")
            .await
            .unwrap();
        assert!(
            result.variants.len() >= 3,
            "Expected at least 3 variants, got {}",
            result.variants.len()
        );
    }

    #[tokio::test]
    async fn does_not_exceed_max_variants() {
        let agent = make_agent();
        let scorecard = make_scorecard(0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4);
        let config = OptimizerConfig {
            min_variants: 2,
            max_variants: 5,
            llm_api_key: "".to_string(),
            ..Default::default()
        };
        let optimizer = Optimizer::new(config);
        let result = optimizer
            .generate_variants(&agent, &scorecard, &[], "abc123")
            .await
            .unwrap();
        assert!(result.variants.len() <= 5);
    }

    #[tokio::test]
    async fn variants_have_non_empty_descriptions() {
        let agent = make_agent();
        let scorecard = make_scorecard(0.6, 0.5, 0.7, 0.7, 0.5, 0.5, 0.7);
        let config = OptimizerConfig {
            min_variants: 2,
            max_variants: 5,
            llm_api_key: "".to_string(),
            ..Default::default()
        };
        let optimizer = Optimizer::new(config);
        let result = optimizer
            .generate_variants(&agent, &scorecard, &[], "sha_x")
            .await
            .unwrap();
        for v in &result.variants {
            assert!(
                !v.description.is_empty(),
                "Variant description must not be empty"
            );
        }
    }

    #[tokio::test]
    async fn all_variants_have_correct_parent_sha() {
        let agent = make_agent();
        let scorecard = make_scorecard(0.6, 0.5, 0.7, 0.7, 0.5, 0.5, 0.7);
        let config = OptimizerConfig {
            min_variants: 3,
            max_variants: 6,
            llm_api_key: "".to_string(),
            ..Default::default()
        };
        let optimizer = Optimizer::new(config);
        let result = optimizer
            .generate_variants(&agent, &scorecard, &[], "parent_sha_42")
            .await
            .unwrap();
        for v in &result.variants {
            assert_eq!(
                v.parent_sha, "parent_sha_42",
                "All variants must reference the correct parent SHA"
            );
        }
    }

    #[tokio::test]
    async fn all_variants_preserve_agent_name() {
        let agent = make_agent();
        let scorecard = make_scorecard(0.6, 0.5, 0.7, 0.7, 0.5, 0.5, 0.7);
        let config = OptimizerConfig {
            min_variants: 2,
            max_variants: 5,
            llm_api_key: "".to_string(),
            ..Default::default()
        };
        let optimizer = Optimizer::new(config);
        let result = optimizer
            .generate_variants(&agent, &scorecard, &[], "sha")
            .await
            .unwrap();
        for v in &result.variants {
            assert_eq!(
                v.agent.name, "test-agent",
                "Agent name must be preserved across variants"
            );
        }
    }

    #[tokio::test]
    async fn low_task_completion_triggers_instruction_reorder_fallback() {
        let agent = make_agent();
        // Low task completion but no LLM key — should fall back to reorder
        let scorecard = make_scorecard(0.5, 0.3, 0.8, 0.8, 0.3, 0.8, 0.8);
        let config = OptimizerConfig {
            min_variants: 1,
            max_variants: 10,
            llm_api_key: "".to_string(),
            ..Default::default()
        };
        let optimizer = Optimizer::new(config);
        let result = optimizer
            .generate_variants(&agent, &scorecard, &[], "sha")
            .await
            .unwrap();
        // At least one variant must be an InstructionReorder (the fallback)
        let has_reorder = result
            .variants
            .iter()
            .any(|v| v.mutation_type == MutationType::InstructionReorder);
        assert!(
            has_reorder,
            "Without LLM, InstructionReorder fallback must be applied"
        );
    }

    #[tokio::test]
    async fn few_shot_injection_skipped_when_not_enough_traces() {
        let agent = make_agent();
        let scorecard = make_scorecard(0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8);
        let config = OptimizerConfig {
            min_variants: 2,
            max_variants: 10,
            llm_api_key: "".to_string(),
            few_shot_min_traces: 50, // requires 50 passing traces
            ..Default::default()
        };
        let optimizer = Optimizer::new(config);
        let result = optimizer
            .generate_variants(&agent, &scorecard, &[], "sha") // passing_traces = []
            .await
            .unwrap();
        let has_few_shot = result
            .mutation_types_applied
            .contains(&MutationType::FewShotInjection);
        assert!(
            !has_few_shot,
            "FewShotInjection must not be applied when insufficient passing traces"
        );
    }

    #[tokio::test]
    async fn min_variants_satisfied_when_all_mutations_skipped() {
        // Pass/high scores so no mutation conditions trigger
        let agent = {
            let mut a = make_agent();
            a.tools = vec![]; // no tools → tool description rewrite skipped
            a.output_schema = None; // no schema → schema tighten skipped
            a
        };
        let scorecard = make_scorecard(0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95);
        let config = OptimizerConfig {
            min_variants: 3,
            max_variants: 10,
            llm_api_key: "".to_string(),
            few_shot_min_traces: 50,
            ..Default::default()
        };
        let optimizer = Optimizer::new(config);
        let result = optimizer
            .generate_variants(&agent, &scorecard, &[], "sha")
            .await
            .unwrap();
        assert!(
            result.variants.len() >= 3,
            "min_variants must be satisfied even when no mutations apply"
        );
    }
}
