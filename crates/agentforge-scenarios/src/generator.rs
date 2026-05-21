use agentforge_core::{AgentFile, Result, Scenario};

use crate::{
    adversarial::generate_adversarial_scenarios,
    domain_seeded::{generate_domain_seeded_scenarios, DomainSeededConfig},
    schema_derived::generate_schema_derived_scenarios,
};

/// Configuration for scenario generation.
#[derive(Debug, Clone)]
pub struct ScenarioGeneratorConfig {
    pub total_count: u32,
    /// Fraction of scenarios generated via schema-derived strategy (default: 0.5)
    pub schema_derived_ratio: f64,
    /// Fraction of scenarios generated adversarially (default: 0.3)
    pub adversarial_ratio: f64,
    /// Fraction of scenarios generated via domain seeding (default: 0.2)
    pub domain_seeded_ratio: f64,
    /// OpenAI-compatible LLM base URL for domain-seeded generation (optional)
    pub llm_base_url: Option<String>,
    pub llm_api_key: Option<String>,
    pub llm_model: Option<String>,
    pub agent_id: uuid::Uuid,
}

impl ScenarioGeneratorConfig {
    /// Validate that ratios sum to approximately 1.0
    pub fn validate(&self) -> bool {
        let total = self.schema_derived_ratio + self.adversarial_ratio + self.domain_seeded_ratio;
        (total - 1.0).abs() < 0.01
    }
}

impl Default for ScenarioGeneratorConfig {
    fn default() -> Self {
        Self {
            total_count: 100,
            schema_derived_ratio: 0.5,
            adversarial_ratio: 0.3,
            domain_seeded_ratio: 0.2,
            llm_base_url: None,
            llm_api_key: None,
            llm_model: None,
            agent_id: uuid::Uuid::new_v4(),
        }
    }
}

/// Generate scenarios for an agent file using all three strategies.
pub async fn generate_scenarios(
    agent: &AgentFile,
    config: &ScenarioGeneratorConfig,
) -> Result<Vec<Scenario>> {
    let total = config.total_count as usize;
    // Use floor so that rounding never eats the domain-seeded budget entirely.
    // With total=5: schema=2, adversarial=1, domain=2.
    let schema_n = (total as f64 * config.schema_derived_ratio).floor() as usize;
    let adversarial_n = (total as f64 * config.adversarial_ratio).floor() as usize;
    let domain_n = total.saturating_sub(schema_n + adversarial_n);

    tracing::info!(
        agent = %agent.name,
        total = total,
        schema_n = schema_n,
        adversarial_n = adversarial_n,
        domain_n = domain_n,
        "Generating scenarios"
    );

    let mut scenarios = Vec::with_capacity(total);

    // Schema-derived scenarios
    let schema_scenarios = generate_schema_derived_scenarios(agent, schema_n, config.agent_id)?;
    scenarios.extend(schema_scenarios);

    // Adversarial scenarios
    let adversarial = generate_adversarial_scenarios(agent, adversarial_n, config.agent_id)?;
    scenarios.extend(adversarial);

    // Domain-seeded scenarios (LLM-based, optional)
    if domain_n > 0 {
        let domain_config = DomainSeededConfig {
            count: domain_n,
            agent_id: config.agent_id,
            llm_base_url: config.llm_base_url.clone(),
            llm_api_key: config.llm_api_key.clone(),
            llm_model: config
                .llm_model
                .clone()
                .unwrap_or_else(|| "gpt-4o-mini".to_string()),
        };

        match generate_domain_seeded_scenarios(agent, &domain_config).await {
            Ok(ds) => scenarios.extend(ds),
            Err(e) => {
                tracing::warn!(error = %e, "Domain-seeded generation failed, falling back to adversarial");
                let fallback = generate_adversarial_scenarios(agent, domain_n, config.agent_id)?;
                scenarios.extend(fallback);
            }
        }
    }

    tracing::info!(generated = scenarios.len(), "Scenario generation complete");
    Ok(scenarios)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_core::{EvalHints, ModelConfig, ModelProvider, ToolDefinition};

    fn make_test_agent() -> AgentFile {
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
            system_prompt: "You are a customer support agent. Help users with orders.".to_string(),
            tools: vec![
                ToolDefinition {
                    name: "get_order_status".to_string(),
                    description: "Get order status by ID".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "Order ID"}
                        },
                        "required": ["order_id"]
                    }),
                },
                ToolDefinition {
                    name: "cancel_order".to_string(),
                    description: "Cancel an order".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"}
                        },
                        "required": ["order_id"]
                    }),
                },
            ],
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "action_taken": {"type": "string"}
                },
                "required": ["response"]
            })),
            constraints: vec![
                "Always confirm order ID before calling get_order_status.".to_string()
            ],
            eval_hints: Some(EvalHints {
                domain: Some("customer_support".to_string()),
                typical_turns: Some(3),
                critical_tools: vec!["get_order_status".to_string()],
                pass_threshold: Some(0.85),
                scenario_count: Some(20),
            }),
            metadata: None,
        }
    }

    #[test]
    fn default_config_valid() {
        let config = ScenarioGeneratorConfig::default();
        assert!(config.validate());
    }

    #[tokio::test]
    async fn generates_expected_count() {
        let agent = make_test_agent();
        let config = ScenarioGeneratorConfig {
            total_count: 10,
            schema_derived_ratio: 0.5,
            adversarial_ratio: 0.3,
            domain_seeded_ratio: 0.2,
            llm_base_url: None,
            llm_api_key: None,
            llm_model: None,
            agent_id: uuid::Uuid::new_v4(),
        };
        let scenarios = generate_scenarios(&agent, &config).await.unwrap();
        assert!(!scenarios.is_empty());
        // Should have at least schema-derived and adversarial
        assert!(scenarios.len() >= 8);
    }
}
