use serde::{Deserialize, Serialize};

/// Score result for a single dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    pub value: f64,
    /// Confidence that this score is correct (0.0–1.0).
    /// Scores below 0.5 confidence are flagged for human review.
    pub confidence: f64,
    pub method: ScoringMethod,
    pub rationale: Option<String>,
}

impl Default for DimensionScore {
    fn default() -> Self {
        Self {
            value: 0.0,
            confidence: 1.0,
            method: ScoringMethod::Deterministic,
            rationale: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoringMethod {
    /// Deterministic check (schema validation, keyword match, etc.)
    Deterministic,
    /// LLM judge evaluation
    LlmJudge,
    /// Combination of both
    Hybrid,
    /// Heuristic approximation (no LLM, no hard rules)
    Heuristic,
}

/// Scorecard: the aggregated scoring result for a full eval run,
/// ready for display in the dashboard / CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    pub run_id: uuid::Uuid,
    pub agent_id: uuid::Uuid,
    pub agent_name: String,
    pub agent_version: String,
    pub aggregate_score: f64,
    pub pass_rate: f64,
    pub total_scenarios: u32,
    pub passed: u32,
    pub failed: u32,
    pub errors: u32,
    pub review_needed: u32,
    pub dimension_scores: crate::DimensionScores,
    pub failure_clusters: Vec<crate::FailureClusterSummary>,
    pub duration_seconds: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

/// A scorecard delta between two agent versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardDiff {
    pub version_a: String,
    pub version_b: String,
    pub aggregate_score_delta: f64,
    pub pass_rate_delta: f64,
    pub dimension_deltas: DimensionDeltas,
    pub regression_count: u32,
    pub improvement_count: u32,
    pub neutral_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionDeltas {
    pub task_completion: f64,
    pub tool_selection: f64,
    pub argument_correctness: f64,
    pub schema_compliance: f64,
    pub instruction_adherence: f64,
    pub path_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_method_serde() {
        let json = serde_json::to_string(&ScoringMethod::LlmJudge).unwrap();
        assert_eq!(json, r#""llm_judge""#);
        let back: ScoringMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ScoringMethod::LlmJudge);
    }

    // ── 12 new tests ─────────────────────────────────────────────────────────

    #[test]
    fn scoring_method_serde_all_variants() {
        let pairs = [
            (ScoringMethod::Deterministic, r#""deterministic""#),
            (ScoringMethod::LlmJudge, r#""llm_judge""#),
            (ScoringMethod::Hybrid, r#""hybrid""#),
            (ScoringMethod::Heuristic, r#""heuristic""#),
        ];
        for (method, expected) in &pairs {
            let json = serde_json::to_string(method).unwrap();
            assert_eq!(
                &json, expected,
                "ScoringMethod::{:?} serialized incorrectly",
                method
            );
            let back: ScoringMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, method);
        }
    }

    #[test]
    fn scoring_method_all_variants_are_distinct() {
        // Each variant serializes to a unique string — use that for uniqueness check
        let all = [
            ScoringMethod::Deterministic,
            ScoringMethod::LlmJudge,
            ScoringMethod::Hybrid,
            ScoringMethod::Heuristic,
        ];
        let strs: std::collections::HashSet<_> = all
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();
        assert_eq!(strs.len(), 4);
    }

    #[test]
    fn dimension_score_default_value_is_zero() {
        let ds = DimensionScore::default();
        assert_eq!(ds.value, 0.0);
        assert_eq!(ds.confidence, 1.0);
        assert_eq!(ds.method, ScoringMethod::Deterministic);
        assert!(ds.rationale.is_none());
    }

    #[test]
    fn dimension_score_stores_rationale() {
        let ds = DimensionScore {
            value: 0.75,
            confidence: 0.9,
            method: ScoringMethod::LlmJudge,
            rationale: Some("Agent called the wrong tool".to_string()),
        };
        assert!((ds.value - 0.75).abs() < 1e-9);
        assert_eq!(ds.rationale.as_deref(), Some("Agent called the wrong tool"));
    }

    #[test]
    fn dimension_score_serde_roundtrip() {
        let ds = DimensionScore {
            value: 0.85,
            confidence: 0.95,
            method: ScoringMethod::Hybrid,
            rationale: Some("Partial pass".to_string()),
        };
        let json = serde_json::to_string(&ds).unwrap();
        let back: DimensionScore = serde_json::from_str(&json).unwrap();
        assert!((back.value - 0.85).abs() < 1e-9);
        assert_eq!(back.method, ScoringMethod::Hybrid);
    }

    #[test]
    fn dimension_score_null_rationale_serde() {
        let ds = DimensionScore {
            value: 1.0,
            confidence: 1.0,
            method: ScoringMethod::Deterministic,
            rationale: None,
        };
        let json = serde_json::to_value(&ds).unwrap();
        assert!(json["rationale"].is_null());
    }

    #[test]
    fn scorecard_diff_stores_delta_fields() {
        let diff = ScorecardDiff {
            version_a: "v1.0.0".to_string(),
            version_b: "v1.1.0".to_string(),
            aggregate_score_delta: 0.05,
            pass_rate_delta: 0.10,
            dimension_deltas: DimensionDeltas {
                task_completion: 0.02,
                tool_selection: 0.03,
                argument_correctness: -0.01,
                schema_compliance: 0.00,
                instruction_adherence: 0.01,
                path_efficiency: 0.00,
            },
            regression_count: 2,
            improvement_count: 5,
            neutral_count: 3,
        };
        assert_eq!(diff.version_a, "v1.0.0");
        assert!((diff.aggregate_score_delta - 0.05).abs() < 1e-9);
        assert_eq!(diff.improvement_count, 5);
    }

    #[test]
    fn scorecard_diff_serde_roundtrip() {
        let diff = ScorecardDiff {
            version_a: "a".to_string(),
            version_b: "b".to_string(),
            aggregate_score_delta: 0.03,
            pass_rate_delta: 0.05,
            dimension_deltas: DimensionDeltas {
                task_completion: 0.01,
                tool_selection: 0.02,
                argument_correctness: 0.00,
                schema_compliance: 0.00,
                instruction_adherence: 0.00,
                path_efficiency: 0.00,
            },
            regression_count: 0,
            improvement_count: 3,
            neutral_count: 7,
        };
        let json = serde_json::to_string(&diff).unwrap();
        let back: ScorecardDiff = serde_json::from_str(&json).unwrap();
        assert_eq!(back.version_b, "b");
        assert!((back.aggregate_score_delta - 0.03).abs() < 1e-9);
    }

    #[test]
    fn dimension_deltas_serde_roundtrip() {
        let d = DimensionDeltas {
            task_completion: 0.1,
            tool_selection: -0.1,
            argument_correctness: 0.0,
            schema_compliance: 0.05,
            instruction_adherence: -0.02,
            path_efficiency: 0.01,
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: DimensionDeltas = serde_json::from_str(&json).unwrap();
        assert!((back.task_completion - 0.1).abs() < 1e-9);
        assert!((back.tool_selection - (-0.1)).abs() < 1e-9);
    }

    #[test]
    fn scorecard_stores_all_fields() {
        let run_id = uuid::Uuid::new_v4();
        let agent_id = uuid::Uuid::new_v4();
        let sc = Scorecard {
            run_id,
            agent_id,
            agent_name: "test".to_string(),
            agent_version: "1.0.0".to_string(),
            aggregate_score: 0.75,
            pass_rate: 0.60,
            total_scenarios: 10,
            passed: 6,
            failed: 3,
            errors: 1,
            review_needed: 0,
            dimension_scores: crate::DimensionScores::default(),
            failure_clusters: vec![],
            duration_seconds: 120,
            total_input_tokens: 5000,
            total_output_tokens: 2000,
        };
        assert_eq!(sc.run_id, run_id);
        assert_eq!(sc.total_scenarios, 10);
        assert_eq!(sc.passed + sc.failed + sc.errors, 10);
    }

    #[test]
    fn scoring_method_eq_deterministic() {
        assert_eq!(ScoringMethod::Deterministic, ScoringMethod::Deterministic);
        assert_ne!(ScoringMethod::Deterministic, ScoringMethod::LlmJudge);
    }

    #[test]
    fn dimension_score_confidence_range_is_valid() {
        // Confidence must be [0, 1]; test boundary values
        let low = DimensionScore {
            value: 0.5,
            confidence: 0.0,
            method: ScoringMethod::Heuristic,
            rationale: None,
        };
        let high = DimensionScore {
            value: 0.5,
            confidence: 1.0,
            method: ScoringMethod::Heuristic,
            rationale: None,
        };
        assert!((0.0..=1.0).contains(&low.confidence));
        assert!((0.0..=1.0).contains(&high.confidence));
    }
}
