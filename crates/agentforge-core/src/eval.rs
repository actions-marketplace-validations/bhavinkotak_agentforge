use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An evaluation run of an agent version against a scenario set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRun {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub scenario_set_id: Option<Uuid>,
    pub status: EvalRunStatus,
    pub scenario_count: u32,
    pub completed_count: u32,
    pub error_count: u32,
    pub aggregate_score: Option<f64>,
    pub pass_rate: Option<f64>,
    pub scores: Option<DimensionScores>,
    pub failure_clusters: Option<Vec<FailureClusterSummary>>,
    pub seed: u32,
    pub concurrency: u32,
    pub error_message: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    // ── Self-improvement loop tracking ────────────────────────────────────────
    /// Optimization loop state: `running` | `converged` | `no_improvement` |
    /// `max_iterations` | `failed`. `None` means optimization was not requested.
    pub opt_status: Option<String>,
    /// Number of completed optimization rounds (0 = not started).
    pub opt_rounds: i32,
    /// Best aggregate score achieved across all optimization rounds.
    pub opt_best_score: Option<f64>,
    /// UUID of the best agent version saved during the optimization loop.
    pub opt_best_agent_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalRunStatus {
    Pending,
    Running,
    Complete,
    Error,
    Cancelled,
}

impl std::fmt::Display for EvalRunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalRunStatus::Pending => write!(f, "pending"),
            EvalRunStatus::Running => write!(f, "running"),
            EvalRunStatus::Complete => write!(f, "complete"),
            EvalRunStatus::Error => write!(f, "error"),
            EvalRunStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Per-dimension score breakdown.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DimensionScores {
    /// Did the agent achieve the goal? Weight: 35%
    pub task_completion: f64,
    /// Were the right tools called? Weight: 20%
    pub tool_selection: f64,
    /// Were tool arguments valid and correct? Weight: 20%
    pub argument_correctness: f64,
    /// Was the output schema compliant? Weight: 15%
    pub schema_compliance: f64,
    /// Did the agent follow constraints? Weight: 7%
    pub instruction_adherence: f64,
    /// Was the shortest valid path taken? Weight: 3%
    pub path_efficiency: f64,
}

impl DimensionScores {
    /// Scoring weights as defined in the PRD (configurable per run via EvalWeights).
    pub fn weighted_aggregate(&self, weights: &EvalWeights) -> f64 {
        self.task_completion * weights.task_completion
            + self.tool_selection * weights.tool_selection
            + self.argument_correctness * weights.argument_correctness
            + self.schema_compliance * weights.schema_compliance
            + self.instruction_adherence * weights.instruction_adherence
            + self.path_efficiency * weights.path_efficiency
    }
}

/// Configurable weights for the 6 evaluation dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalWeights {
    pub task_completion: f64,
    pub tool_selection: f64,
    pub argument_correctness: f64,
    pub schema_compliance: f64,
    pub instruction_adherence: f64,
    pub path_efficiency: f64,
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self {
            task_completion: 0.35,
            tool_selection: 0.20,
            argument_correctness: 0.20,
            schema_compliance: 0.15,
            instruction_adherence: 0.07,
            path_efficiency: 0.03,
        }
    }
}

impl EvalWeights {
    /// Validate that all weights sum to approximately 1.0.
    pub fn validate(&self) -> bool {
        let total = self.task_completion
            + self.tool_selection
            + self.argument_correctness
            + self.schema_compliance
            + self.instruction_adherence
            + self.path_efficiency;
        (total - 1.0).abs() < 0.001
    }
}

/// Summary of failure clusters across a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureClusterSummary {
    pub cluster: FailureCluster,
    pub count: u32,
    pub percentage: f64,
    pub sample_scenarios: Vec<Uuid>,
}

/// Root cause categories for trace failures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FailureCluster {
    WrongTool,
    HallucinatedArgument,
    Looping,
    PrematureStop,
    SchemaViolation,
    ConstraintBreach,
    NoFailure,
    /// Trace failed due to an LLM/API infrastructure error (rate limit, timeout,
    /// 5xx), not due to agent behaviour. These do not reflect agent quality.
    ApiError,
    Unknown,
}

impl std::fmt::Display for FailureCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailureCluster::WrongTool => write!(f, "wrong_tool"),
            FailureCluster::HallucinatedArgument => write!(f, "hallucinated_argument"),
            FailureCluster::Looping => write!(f, "looping"),
            FailureCluster::PrematureStop => write!(f, "premature_stop"),
            FailureCluster::SchemaViolation => write!(f, "schema_violation"),
            FailureCluster::ConstraintBreach => write!(f, "constraint_breach"),
            FailureCluster::NoFailure => write!(f, "no_failure"),
            FailureCluster::ApiError => write!(f, "api_error"),
            FailureCluster::Unknown => write!(f, "unknown"),
        }
    }
}

impl EvalRun {
    /// Convert a completed EvalRun to a Scorecard for display/gatekeeper use.
    /// Returns None if the run has no scores yet.
    pub fn to_scorecard(&self) -> Option<crate::Scorecard> {
        let scores = self.scores.clone()?;
        let aggregate_score = self.aggregate_score?;
        let pass_rate = self.pass_rate?;
        Some(crate::Scorecard {
            run_id: self.id,
            agent_id: self.agent_id,
            agent_name: String::new(), // filled by caller if needed
            agent_version: String::new(),
            aggregate_score,
            pass_rate,
            total_scenarios: self.scenario_count,
            passed: (pass_rate * self.scenario_count as f64) as u32,
            failed: self.error_count,
            errors: 0,
            review_needed: 0,
            dimension_scores: scores,
            failure_clusters: self.failure_clusters.clone().unwrap_or_default(),
            duration_seconds: self
                .completed_at
                .zip(self.started_at)
                .map(|(c, s)| (c - s).num_seconds().max(0) as u64)
                .unwrap_or(0),
            total_input_tokens: 0,
            total_output_tokens: 0,
        })
    }
}

/// Request to start a new eval run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRunRequest {
    pub agent_id: Uuid,
    pub scenario_count: Option<u32>,
    pub concurrency: Option<u32>,
    pub seed: Option<u32>,
    pub weights: Option<EvalWeights>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_weights_sum_to_one() {
        let weights = EvalWeights::default();
        assert!(weights.validate(), "Default weights must sum to 1.0");
    }

    #[test]
    fn weighted_aggregate_perfect_score() {
        let scores = DimensionScores {
            task_completion: 1.0,
            tool_selection: 1.0,
            argument_correctness: 1.0,
            schema_compliance: 1.0,
            instruction_adherence: 1.0,
            path_efficiency: 1.0,
        };
        let weights = EvalWeights::default();
        let agg = scores.weighted_aggregate(&weights);
        assert!((agg - 1.0).abs() < 1e-9);
    }

    #[test]
    fn weighted_aggregate_zero_score() {
        let scores = DimensionScores::default();
        let weights = EvalWeights::default();
        assert_eq!(scores.weighted_aggregate(&weights), 0.0);
    }

    #[test]
    fn failure_cluster_display() {
        assert_eq!(FailureCluster::WrongTool.to_string(), "wrong_tool");
        assert_eq!(
            FailureCluster::HallucinatedArgument.to_string(),
            "hallucinated_argument"
        );
    }

    // ── 20 new tests below ───────────────────────────────────────────────────

    #[test]
    fn weights_that_do_not_sum_to_one_fail_validate() {
        let bad = EvalWeights {
            task_completion: 0.5,
            tool_selection: 0.4,
            argument_correctness: 0.0,
            schema_compliance: 0.0,
            instruction_adherence: 0.0,
            path_efficiency: 0.0,
        };
        assert!(!bad.validate());
    }

    #[test]
    fn weights_summing_to_exactly_one_are_valid() {
        let w = EvalWeights {
            task_completion: 0.35,
            tool_selection: 0.20,
            argument_correctness: 0.20,
            schema_compliance: 0.15,
            instruction_adherence: 0.07,
            path_efficiency: 0.03,
        };
        assert!(w.validate());
    }

    #[test]
    fn weighted_aggregate_only_task_completion() {
        let scores = DimensionScores {
            task_completion: 1.0,
            ..DimensionScores::default()
        };
        let weights = EvalWeights::default();
        // Only task_completion=1.0 contributes → 0.35
        let agg = scores.weighted_aggregate(&weights);
        assert!((agg - 0.35).abs() < 1e-9);
    }

    #[test]
    fn weighted_aggregate_only_tool_selection() {
        let scores = DimensionScores {
            tool_selection: 1.0,
            ..DimensionScores::default()
        };
        let weights = EvalWeights::default();
        let agg = scores.weighted_aggregate(&weights);
        assert!((agg - 0.20).abs() < 1e-9);
    }

    #[test]
    fn weighted_aggregate_only_schema_compliance() {
        let scores = DimensionScores {
            schema_compliance: 1.0,
            ..DimensionScores::default()
        };
        let weights = EvalWeights::default();
        let agg = scores.weighted_aggregate(&weights);
        assert!((agg - 0.15).abs() < 1e-9);
    }

    #[test]
    fn weighted_aggregate_respects_custom_weights() {
        let scores = DimensionScores {
            task_completion: 1.0,
            ..DimensionScores::default()
        };
        let weights = EvalWeights {
            task_completion: 1.0,
            tool_selection: 0.0,
            argument_correctness: 0.0,
            schema_compliance: 0.0,
            instruction_adherence: 0.0,
            path_efficiency: 0.0,
        };
        let agg = scores.weighted_aggregate(&weights);
        assert!((agg - 1.0).abs() < 1e-9);
    }

    #[test]
    fn eval_run_status_display_all_variants() {
        assert_eq!(EvalRunStatus::Pending.to_string(), "pending");
        assert_eq!(EvalRunStatus::Running.to_string(), "running");
        assert_eq!(EvalRunStatus::Complete.to_string(), "complete");
        assert_eq!(EvalRunStatus::Error.to_string(), "error");
        assert_eq!(EvalRunStatus::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn eval_run_status_all_variants_distinct() {
        let all = [
            EvalRunStatus::Pending.to_string(),
            EvalRunStatus::Running.to_string(),
            EvalRunStatus::Complete.to_string(),
            EvalRunStatus::Error.to_string(),
            EvalRunStatus::Cancelled.to_string(),
        ];
        let set: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(set.len(), 5, "All status strings must be distinct");
    }

    #[test]
    fn failure_cluster_display_all_variants() {
        assert_eq!(FailureCluster::WrongTool.to_string(), "wrong_tool");
        assert_eq!(
            FailureCluster::HallucinatedArgument.to_string(),
            "hallucinated_argument"
        );
        assert_eq!(FailureCluster::Looping.to_string(), "looping");
        assert_eq!(FailureCluster::PrematureStop.to_string(), "premature_stop");
        assert_eq!(
            FailureCluster::SchemaViolation.to_string(),
            "schema_violation"
        );
        assert_eq!(
            FailureCluster::ConstraintBreach.to_string(),
            "constraint_breach"
        );
        assert_eq!(FailureCluster::NoFailure.to_string(), "no_failure");
        assert_eq!(FailureCluster::ApiError.to_string(), "api_error");
        assert_eq!(FailureCluster::Unknown.to_string(), "unknown");
    }

    #[test]
    fn failure_cluster_serde_roundtrip() {
        let original = FailureCluster::HallucinatedArgument;
        let json = serde_json::to_string(&original).unwrap();
        let back: FailureCluster = serde_json::from_str(&json).unwrap();
        assert_eq!(back, FailureCluster::HallucinatedArgument);
    }

    #[test]
    fn failure_cluster_api_error_serde() {
        let cluster = FailureCluster::ApiError;
        let json = serde_json::to_string(&cluster).unwrap();
        assert_eq!(json, r#""api_error""#);
        let back: FailureCluster = serde_json::from_str(&json).unwrap();
        assert_eq!(back, FailureCluster::ApiError);
    }

    #[test]
    fn dimension_scores_default_are_zero() {
        let s = DimensionScores::default();
        assert_eq!(s.task_completion, 0.0);
        assert_eq!(s.tool_selection, 0.0);
        assert_eq!(s.argument_correctness, 0.0);
        assert_eq!(s.schema_compliance, 0.0);
        assert_eq!(s.instruction_adherence, 0.0);
        assert_eq!(s.path_efficiency, 0.0);
    }

    #[test]
    fn dimension_scores_serde_roundtrip() {
        let s = DimensionScores {
            task_completion: 0.9,
            tool_selection: 0.8,
            argument_correctness: 0.7,
            schema_compliance: 0.6,
            instruction_adherence: 0.5,
            path_efficiency: 0.4,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: DimensionScores = serde_json::from_str(&json).unwrap();
        assert!((back.task_completion - 0.9).abs() < 1e-9);
        assert!((back.path_efficiency - 0.4).abs() < 1e-9);
    }

    #[test]
    fn eval_run_status_serde_snake_case() {
        let json = serde_json::to_string(&EvalRunStatus::Complete).unwrap();
        assert_eq!(json, r#""complete""#);
        let back: EvalRunStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, EvalRunStatus::Complete);
    }

    #[test]
    fn eval_run_status_pending_serde() {
        let json = serde_json::to_string(&EvalRunStatus::Pending).unwrap();
        assert_eq!(json, r#""pending""#);
    }

    #[test]
    fn eval_run_status_cancelled_serde() {
        let json = serde_json::to_string(&EvalRunStatus::Cancelled).unwrap();
        assert_eq!(json, r#""cancelled""#);
    }

    #[test]
    fn weights_tolerance_accepts_floating_point_imprecision() {
        // When weights are computed programmatically, floating point may cause
        // tiny deviations from 1.0 — the 0.001 tolerance covers this.
        let w = EvalWeights {
            task_completion: 0.35 + 0.000001,
            tool_selection: 0.20,
            argument_correctness: 0.20,
            schema_compliance: 0.15,
            instruction_adherence: 0.07,
            path_efficiency: 0.03,
        };
        // Sum is 1.000001 — within tolerance
        assert!(w.validate());
    }

    #[test]
    fn failure_cluster_summary_stores_fields() {
        let id = Uuid::new_v4();
        let s = FailureClusterSummary {
            cluster: FailureCluster::Looping,
            count: 3,
            percentage: 30.0,
            sample_scenarios: vec![id],
        };
        assert_eq!(s.count, 3);
        assert!((s.percentage - 30.0).abs() < 1e-9);
        assert_eq!(s.sample_scenarios[0], id);
    }

    #[test]
    fn failure_cluster_all_variants_are_hash_compatible() {
        let mut map = std::collections::HashMap::new();
        map.insert(FailureCluster::WrongTool, 1u32);
        map.insert(FailureCluster::Looping, 2u32);
        map.insert(FailureCluster::ApiError, 3u32);
        assert_eq!(map.get(&FailureCluster::WrongTool), Some(&1));
        assert_eq!(map.get(&FailureCluster::ApiError), Some(&3));
    }

    #[test]
    fn eval_run_request_serde_roundtrip() {
        let req = EvalRunRequest {
            agent_id: Uuid::new_v4(),
            scenario_count: Some(50),
            concurrency: Some(5),
            seed: Some(99),
            weights: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: EvalRunRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.scenario_count, Some(50));
        assert_eq!(back.seed, Some(99));
    }
}
