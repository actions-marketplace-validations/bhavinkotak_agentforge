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
}
