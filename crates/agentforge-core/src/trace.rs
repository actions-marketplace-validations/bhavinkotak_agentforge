use crate::{DimensionScores, FailureCluster};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A complete execution trace for a single scenario run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub id: Uuid,
    pub run_id: Uuid,
    pub scenario_id: Uuid,
    pub status: TraceStatus,
    /// Ordered list of execution steps.
    pub steps: Vec<TraceStep>,
    pub final_output: Option<serde_json::Value>,
    pub scores: Option<DimensionScores>,
    pub aggregate_score: Option<f64>,
    pub failure_cluster: FailureCluster,
    pub failure_reason: Option<String>,
    pub review_needed: bool,
    pub llm_calls: u32,
    pub tool_invocations: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    pub retry_count: u32,
    pub seed: u32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    Pass,
    Fail,
    Error,
    ReviewNeeded,
}

impl std::fmt::Display for TraceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceStatus::Pass => write!(f, "pass"),
            TraceStatus::Fail => write!(f, "fail"),
            TraceStatus::Error => write!(f, "error"),
            TraceStatus::ReviewNeeded => write!(f, "review_needed"),
        }
    }
}

/// A single step in an execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TraceStep {
    LlmCall(LlmCallStep),
    ToolCall(ToolCallStep),
    ToolResult(ToolResultStep),
    AgentThought(AgentThoughtStep),
    FinalOutput(FinalOutputStep),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallStep {
    pub index: u32,
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    pub response: serde_json::Value,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallStep {
    pub index: u32,
    pub tool_name: String,
    pub call_id: String,
    pub arguments: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultStep {
    pub index: u32,
    pub tool_name: String,
    pub call_id: String,
    pub result: serde_json::Value,
    pub is_error: bool,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentThoughtStep {
    pub index: u32,
    pub thought: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalOutputStep {
    pub index: u32,
    pub output: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_status_display() {
        assert_eq!(TraceStatus::Pass.to_string(), "pass");
        assert_eq!(TraceStatus::ReviewNeeded.to_string(), "review_needed");
    }

    // ── 10 new tests ─────────────────────────────────────────────────────────

    #[test]
    fn trace_status_display_all_variants() {
        assert_eq!(TraceStatus::Pass.to_string(), "pass");
        assert_eq!(TraceStatus::Fail.to_string(), "fail");
        assert_eq!(TraceStatus::Error.to_string(), "error");
        assert_eq!(TraceStatus::ReviewNeeded.to_string(), "review_needed");
    }

    #[test]
    fn trace_status_serde_roundtrip() {
        for status in &[
            TraceStatus::Pass,
            TraceStatus::Fail,
            TraceStatus::Error,
            TraceStatus::ReviewNeeded,
        ] {
            let json = serde_json::to_string(status).unwrap();
            let back: TraceStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, status);
        }
    }

    #[test]
    fn trace_status_all_variants_distinct() {
        let all = [
            TraceStatus::Pass.to_string(),
            TraceStatus::Fail.to_string(),
            TraceStatus::Error.to_string(),
            TraceStatus::ReviewNeeded.to_string(),
        ];
        let set: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn tool_call_step_serde_roundtrip() {
        let step = ToolCallStep {
            index: 0,
            tool_name: "search".to_string(),
            call_id: "call_abc".to_string(),
            arguments: serde_json::json!({"query": "rust"}),
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&step).unwrap();
        let back: ToolCallStep = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_name, "search");
        assert_eq!(back.call_id, "call_abc");
    }

    #[test]
    fn llm_call_step_serde_roundtrip() {
        let step = LlmCallStep {
            index: 1,
            model: "gpt-4o".to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": "hello"})],
            response: serde_json::json!({"content": "hi"}),
            input_tokens: 10,
            output_tokens: 5,
            latency_ms: 300,
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&step).unwrap();
        let back: LlmCallStep = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "gpt-4o");
        assert_eq!(back.input_tokens, 10);
    }

    #[test]
    fn tool_result_step_error_flag() {
        let step = ToolResultStep {
            index: 2,
            tool_name: "search".to_string(),
            call_id: "call_abc".to_string(),
            result: serde_json::json!({"error": "timeout"}),
            is_error: true,
            timestamp: chrono::Utc::now(),
        };
        assert!(step.is_error);
    }

    #[test]
    fn agent_thought_step_stores_thought() {
        let step = AgentThoughtStep {
            index: 3,
            thought: "I should use the search tool".to_string(),
            timestamp: chrono::Utc::now(),
        };
        assert_eq!(step.thought, "I should use the search tool");
    }

    #[test]
    fn trace_step_tag_type_in_json() {
        // Serde tag="type" means the JSON must include a "type" key
        let step = TraceStep::ToolCall(ToolCallStep {
            index: 0,
            tool_name: "get_order".to_string(),
            call_id: "c1".to_string(),
            arguments: serde_json::json!({}),
            timestamp: chrono::Utc::now(),
        });
        let json = serde_json::to_value(&step).unwrap();
        assert_eq!(json["type"], "tool_call");
        assert_eq!(json["tool_name"], "get_order");
    }

    #[test]
    fn llm_call_step_tag_in_json() {
        let step = TraceStep::LlmCall(LlmCallStep {
            index: 0,
            model: "gpt-4o".to_string(),
            messages: vec![],
            response: serde_json::json!({}),
            input_tokens: 0,
            output_tokens: 0,
            latency_ms: 0,
            timestamp: chrono::Utc::now(),
        });
        let json = serde_json::to_value(&step).unwrap();
        assert_eq!(json["type"], "llm_call");
    }

    #[test]
    fn final_output_step_tag_in_json() {
        let step = TraceStep::FinalOutput(FinalOutputStep {
            index: 5,
            output: serde_json::json!({"result": "done"}),
            timestamp: chrono::Utc::now(),
        });
        let json = serde_json::to_value(&step).unwrap();
        assert_eq!(json["type"], "final_output");
        assert_eq!(json["output"]["result"], "done");
    }
}
