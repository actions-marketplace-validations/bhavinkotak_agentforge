use crate::db_err;
use agentforge_core::{
    AgentForgeError, DimensionScores, FailureCluster, Result, Trace, TraceStatus, TraceStep,
};
use chrono::Utc;
use sqlx::PgPool;
use uuid::Uuid;

pub struct TraceRepo {
    pool: PgPool,
}

impl TraceRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn insert(&self, trace: &Trace) -> Result<()> {
        let status_str = trace.status.to_string();
        let cluster_str = trace.failure_cluster.to_string();
        let steps_json = serde_json::to_value(&trace.steps)
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;
        let scores_json = trace
            .scores
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT INTO traces
                (id, run_id, scenario_id, status, steps, final_output, scores,
                 aggregate_score, failure_cluster, failure_reason, review_needed,
                 llm_calls, tool_invocations, input_tokens, output_tokens,
                 latency_ms, retry_count, seed, created_at)
            VALUES
                ($1, $2, $3, $4::trace_status, $5, $6, $7, $8,
                 $9::failure_cluster, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            "#,
        )
        .bind(trace.id)
        .bind(trace.run_id)
        .bind(trace.scenario_id)
        .bind(status_str)
        .bind(steps_json)
        .bind(trace.final_output.clone())
        .bind(scores_json)
        .bind(trace.aggregate_score)
        .bind(cluster_str)
        .bind(trace.failure_reason.clone())
        .bind(trace.review_needed)
        .bind(trace.llm_calls as i32)
        .bind(trace.tool_invocations as i32)
        .bind(trace.input_tokens as i32)
        .bind(trace.output_tokens as i32)
        .bind(trace.latency_ms as i32)
        .bind(trace.retry_count as i32)
        .bind(trace.seed as i32)
        .bind(Utc::now())
        .execute(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(())
    }

    pub async fn find_by_id(&self, id: Uuid) -> Result<Trace> {
        let r = sqlx::query!(
            r#"
            SELECT id, run_id, scenario_id,
                   status as "status: String",
                   steps, final_output, scores, aggregate_score,
                   failure_cluster as "failure_cluster: String",
                   failure_reason, review_needed,
                   llm_calls, tool_invocations, input_tokens, output_tokens,
                   latency_ms, retry_count, seed, created_at
            FROM traces WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(db_err)?
        .ok_or_else(|| AgentForgeError::NotFound {
            resource: "Trace",
            id: id.to_string(),
        })?;

        self.convert_row(
            r.id,
            r.run_id,
            r.scenario_id,
            r.status,
            r.steps,
            r.final_output,
            r.scores,
            r.aggregate_score,
            r.failure_cluster,
            r.failure_reason,
            r.review_needed,
            r.llm_calls,
            r.tool_invocations,
            r.input_tokens,
            r.output_tokens,
            r.latency_ms,
            r.retry_count,
            r.seed,
            r.created_at,
        )
    }

    pub async fn list_by_run(&self, run_id: Uuid) -> Result<Vec<Trace>> {
        let rows = sqlx::query!(
            r#"
            SELECT id, run_id, scenario_id,
                   status as "status: String",
                   steps, final_output, scores, aggregate_score,
                   failure_cluster as "failure_cluster: String",
                   failure_reason, review_needed,
                   llm_calls, tool_invocations, input_tokens, output_tokens,
                   latency_ms, retry_count, seed, created_at
            FROM traces WHERE run_id = $1 ORDER BY created_at ASC
            "#,
            run_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(db_err)?;

        rows.into_iter()
            .map(|r| {
                self.convert_row(
                    r.id,
                    r.run_id,
                    r.scenario_id,
                    r.status,
                    r.steps,
                    r.final_output,
                    r.scores,
                    r.aggregate_score,
                    r.failure_cluster,
                    r.failure_reason,
                    r.review_needed,
                    r.llm_calls,
                    r.tool_invocations,
                    r.input_tokens,
                    r.output_tokens,
                    r.latency_ms,
                    r.retry_count,
                    r.seed,
                    r.created_at,
                )
            })
            .collect()
    }

    /// Paginated version of list_by_run.
    pub async fn list_by_run_paginated(
        &self,
        run_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Trace>> {
        let all = self.list_by_run(run_id).await?;
        let offset = offset.max(0) as usize;
        let limit = limit.max(0) as usize;
        Ok(all.into_iter().skip(offset).take(limit).collect())
    }

    /// Returns all scenario IDs that passed in a given run.
    pub async fn list_passing_scenario_ids(&self, run_id: Uuid) -> Result<Vec<Uuid>> {
        let rows = sqlx::query!(
            "SELECT scenario_id FROM traces WHERE run_id = $1 AND status = 'pass'::trace_status",
            run_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(rows.into_iter().map(|r| r.scenario_id).collect())
    }

    pub async fn count_review_needed(&self, run_id: Uuid) -> Result<i64> {
        let row = sqlx::query!(
            "SELECT COUNT(*) as cnt FROM traces WHERE run_id = $1 AND review_needed = TRUE",
            run_id
        )
        .fetch_one(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(row.cnt.unwrap_or(0))
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_row(
        &self,
        id: Uuid,
        run_id: Uuid,
        scenario_id: Uuid,
        status: String,
        steps: serde_json::Value,
        final_output: Option<serde_json::Value>,
        scores: Option<serde_json::Value>,
        aggregate_score: Option<f64>,
        failure_cluster: String,
        failure_reason: Option<String>,
        review_needed: bool,
        llm_calls: i32,
        tool_invocations: i32,
        input_tokens: i32,
        output_tokens: i32,
        latency_ms: i32,
        retry_count: i32,
        seed: i32,
        created_at: chrono::DateTime<Utc>,
    ) -> Result<Trace> {
        let steps: Vec<TraceStep> = serde_json::from_value(steps)
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;
        let scores: Option<DimensionScores> = scores
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;

        Ok(Trace {
            id,
            run_id,
            scenario_id,
            status: parse_trace_status(&status),
            steps,
            final_output,
            scores,
            aggregate_score,
            failure_cluster: parse_failure_cluster(&failure_cluster),
            failure_reason,
            review_needed,
            llm_calls: llm_calls as u32,
            tool_invocations: tool_invocations as u32,
            input_tokens: input_tokens as u32,
            output_tokens: output_tokens as u32,
            latency_ms: latency_ms as u64,
            retry_count: retry_count as u32,
            seed: seed as u32,
            created_at,
        })
    }
}

fn parse_trace_status(s: &str) -> TraceStatus {
    match s {
        "pass" => TraceStatus::Pass,
        "fail" => TraceStatus::Fail,
        "review_needed" => TraceStatus::ReviewNeeded,
        _ => TraceStatus::Error,
    }
}

fn parse_failure_cluster(s: &str) -> FailureCluster {
    match s {
        "wrong_tool" => FailureCluster::WrongTool,
        "hallucinated_argument" => FailureCluster::HallucinatedArgument,
        "looping" => FailureCluster::Looping,
        "premature_stop" => FailureCluster::PrematureStop,
        "schema_violation" => FailureCluster::SchemaViolation,
        "constraint_breach" => FailureCluster::ConstraintBreach,
        "no_failure" => FailureCluster::NoFailure,
        _ => FailureCluster::Unknown,
    }
}
