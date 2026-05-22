use serde::{Deserialize, Serialize};

/// Token usage for a single trace or model call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn new(input: u32, output: u32) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
        }
    }
}

/// USD cost breakdown for a trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Estimated total cost in USD.
    pub total_usd: f64,
    /// Cost attributed to input tokens.
    pub input_usd: f64,
    /// Cost attributed to output tokens.
    pub output_usd: f64,
    /// Model that generated this cost.
    pub model: String,
    /// Provider (openai, anthropic, etc.)
    pub provider: String,
}

/// A cost optimization recommendation: downgrade a model for a scenario subset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecommendation {
    /// Current model in use.
    pub current_model: String,
    /// Recommended cheaper model.
    pub recommended_model: String,
    /// Estimated savings per 1000 scenarios (USD).
    pub estimated_savings_usd: f64,
    /// Fraction of scenarios where the cheaper model scored equivalently.
    pub equivalent_score_fraction: f64,
    /// Aggregate score of recommended model on the test scenarios.
    pub candidate_aggregate_score: f64,
    /// Aggregate score of current model on the same scenarios.
    pub current_aggregate_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_total() {
        let u = TokenUsage::new(100, 50);
        assert_eq!(u.total_tokens, 150);
    }

    // ── 12 new tests ─────────────────────────────────────────────────────────

    #[test]
    fn token_usage_zero_inputs() {
        let u = TokenUsage::new(0, 0);
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.total_tokens, 0);
    }

    #[test]
    fn token_usage_default_is_zero() {
        let u = TokenUsage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.total_tokens, 0);
    }

    #[test]
    fn token_usage_large_values() {
        let u = TokenUsage::new(1_000_000, 500_000);
        assert_eq!(u.total_tokens, 1_500_000);
    }

    #[test]
    fn token_usage_input_only() {
        let u = TokenUsage::new(200, 0);
        assert_eq!(u.total_tokens, 200);
        assert_eq!(u.output_tokens, 0);
    }

    #[test]
    fn token_usage_output_only() {
        let u = TokenUsage::new(0, 300);
        assert_eq!(u.total_tokens, 300);
        assert_eq!(u.input_tokens, 0);
    }

    #[test]
    fn token_usage_serde_roundtrip() {
        let u = TokenUsage::new(128, 64);
        let json = serde_json::to_string(&u).unwrap();
        let back: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.input_tokens, 128);
        assert_eq!(back.output_tokens, 64);
        assert_eq!(back.total_tokens, 192);
    }

    #[test]
    fn cost_breakdown_default_is_zero() {
        let cb = CostBreakdown::default();
        assert_eq!(cb.total_usd, 0.0);
        assert_eq!(cb.input_usd, 0.0);
        assert_eq!(cb.output_usd, 0.0);
        assert!(cb.model.is_empty());
        assert!(cb.provider.is_empty());
    }

    #[test]
    fn cost_breakdown_stores_fields() {
        let cb = CostBreakdown {
            total_usd: 0.05,
            input_usd: 0.02,
            output_usd: 0.03,
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
        };
        assert!((cb.total_usd - 0.05).abs() < 1e-9);
        assert_eq!(cb.model, "gpt-4o");
        assert_eq!(cb.provider, "openai");
    }

    #[test]
    fn cost_breakdown_serde_roundtrip() {
        let cb = CostBreakdown {
            total_usd: 0.123,
            input_usd: 0.100,
            output_usd: 0.023,
            model: "claude-3".to_string(),
            provider: "anthropic".to_string(),
        };
        let json = serde_json::to_string(&cb).unwrap();
        let back: CostBreakdown = serde_json::from_str(&json).unwrap();
        assert!((back.total_usd - 0.123).abs() < 1e-9);
        assert_eq!(back.provider, "anthropic");
    }

    #[test]
    fn cost_recommendation_stores_all_fields() {
        let rec = CostRecommendation {
            current_model: "gpt-4o".to_string(),
            recommended_model: "gpt-4o-mini".to_string(),
            estimated_savings_usd: 1.5,
            equivalent_score_fraction: 0.95,
            candidate_aggregate_score: 0.82,
            current_aggregate_score: 0.84,
        };
        assert_eq!(rec.current_model, "gpt-4o");
        assert_eq!(rec.recommended_model, "gpt-4o-mini");
        assert!((rec.equivalent_score_fraction - 0.95).abs() < 1e-9);
    }

    #[test]
    fn cost_recommendation_serde_roundtrip() {
        let rec = CostRecommendation {
            current_model: "gpt-4o".to_string(),
            recommended_model: "gpt-4o-mini".to_string(),
            estimated_savings_usd: 2.0,
            equivalent_score_fraction: 0.90,
            candidate_aggregate_score: 0.80,
            current_aggregate_score: 0.83,
        };
        let json = serde_json::to_string(&rec).unwrap();
        let back: CostRecommendation = serde_json::from_str(&json).unwrap();
        assert!((back.estimated_savings_usd - 2.0).abs() < 1e-9);
        assert_eq!(back.recommended_model, "gpt-4o-mini");
    }

    #[test]
    fn token_usage_new_sets_fields_correctly() {
        let u = TokenUsage::new(512, 256);
        assert_eq!(u.input_tokens, 512);
        assert_eq!(u.output_tokens, 256);
        assert_eq!(u.total_tokens, 768);
    }
}
