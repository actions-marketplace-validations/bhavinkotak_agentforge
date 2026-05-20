use crate::db_err;
use agentforge_core::{
    AgentForgeError, DimensionScores, EvalRun, EvalRunStatus, FailureClusterSummary, Result,
};
use chrono::Utc;
use sqlx::PgPool;
use uuid::Uuid;

pub struct EvalRepo {
    pool: PgPool,
}

impl EvalRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn insert(&self, run: &EvalRun) -> Result<EvalRun> {
        let status_str = run.status.to_string();
        let _clusters_json = run
            .failure_clusters
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT INTO eval_runs
                (id, agent_id, scenario_set_id, status, scenario_count,
                 completed_count, error_count, seed, concurrency, created_at, updated_at)
            VALUES ($1, $2, $3, $4::eval_run_status, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(run.id)
        .bind(run.agent_id)
        .bind(run.scenario_set_id)
        .bind(status_str)
        .bind(run.scenario_count as i32)
        .bind(run.completed_count as i32)
        .bind(run.error_count as i32)
        .bind(run.seed as i32)
        .bind(run.concurrency as i32)
        .bind(Utc::now())
        .bind(Utc::now())
        .execute(&self.pool)
        .await
        .map_err(db_err)?;

        self.find_by_id(run.id).await
    }

    pub async fn find_by_id(&self, id: Uuid) -> Result<EvalRun> {
        let row = sqlx::query!(
            r#"
            SELECT id, agent_id, scenario_set_id,
                   status as "status: String",
                   scenario_count, completed_count, error_count,
                   aggregate_score, pass_rate,
                   task_completion, tool_selection, argument_correctness,
                   path_efficiency, schema_compliance, instruction_adherence,
                   failure_clusters, seed, concurrency, error_message,
                   started_at, completed_at, created_at, updated_at
            FROM eval_runs WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(db_err)?
        .ok_or_else(|| AgentForgeError::NotFound {
            resource: "EvalRun",
            id: id.to_string(),
        })?;

        let scores = if let (Some(tc), Some(ts), Some(ac), Some(pe), Some(sc), Some(ia)) = (
            row.task_completion,
            row.tool_selection,
            row.argument_correctness,
            row.path_efficiency,
            row.schema_compliance,
            row.instruction_adherence,
        ) {
            Some(DimensionScores {
                task_completion: tc,
                tool_selection: ts,
                argument_correctness: ac,
                path_efficiency: pe,
                schema_compliance: sc,
                instruction_adherence: ia,
            })
        } else {
            None
        };

        let failure_clusters: Option<Vec<FailureClusterSummary>> = row
            .failure_clusters
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;

        Ok(EvalRun {
            id: row.id,
            agent_id: row.agent_id,
            scenario_set_id: row.scenario_set_id,
            status: parse_status(&row.status),
            scenario_count: row.scenario_count as u32,
            completed_count: row.completed_count as u32,
            error_count: row.error_count as u32,
            aggregate_score: row.aggregate_score,
            pass_rate: row.pass_rate,
            scores,
            failure_clusters,
            seed: row.seed as u32,
            concurrency: row.concurrency as u32,
            error_message: row.error_message,
            started_at: row.started_at,
            completed_at: row.completed_at,
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }

    pub async fn update_status(&self, id: Uuid, status: &EvalRunStatus) -> Result<()> {
        let status_str = status.to_string();
        let started_at = if *status == EvalRunStatus::Running {
            Some(Utc::now())
        } else {
            None
        };
        let completed_at = if matches!(
            status,
            EvalRunStatus::Complete | EvalRunStatus::Error | EvalRunStatus::Cancelled
        ) {
            Some(Utc::now())
        } else {
            None
        };

        sqlx::query(
            r#"
            UPDATE eval_runs
            SET status = $2::eval_run_status,
                started_at = COALESCE($3, started_at),
                completed_at = COALESCE($4, completed_at),
                updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(status_str)
        .bind(started_at)
        .bind(completed_at)
        .execute(&self.pool)
        .await
        .map_err(db_err)?;

        Ok(())
    }

    pub async fn update_progress(&self, id: Uuid, completed: u32, errors: u32) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE eval_runs
            SET completed_count = $2, error_count = $3, updated_at = NOW()
            WHERE id = $1
            "#,
            id,
            completed as i32,
            errors as i32,
        )
        .execute(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(())
    }

    pub async fn save_scores(
        &self,
        id: Uuid,
        scores: &DimensionScores,
        aggregate_score: f64,
        pass_rate: f64,
        failure_clusters: &[FailureClusterSummary],
    ) -> Result<()> {
        let clusters_json = serde_json::to_value(failure_clusters)
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;

        sqlx::query!(
            r#"
            UPDATE eval_runs
            SET aggregate_score = $2, pass_rate = $3,
                task_completion = $4, tool_selection = $5,
                argument_correctness = $6, path_efficiency = $7,
                schema_compliance = $8, instruction_adherence = $9,
                failure_clusters = $10, updated_at = NOW()
            WHERE id = $1
            "#,
            id,
            aggregate_score,
            pass_rate,
            scores.task_completion,
            scores.tool_selection,
            scores.argument_correctness,
            scores.path_efficiency,
            scores.schema_compliance,
            scores.instruction_adherence,
            clusters_json,
        )
        .execute(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(())
    }

    pub async fn list_by_agent(&self, agent_id: Uuid, limit: i64) -> Result<Vec<EvalRun>> {
        let rows = sqlx::query!(
            r#"
            SELECT id FROM eval_runs
            WHERE agent_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
            agent_id,
            limit,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(db_err)?;

        let mut results = Vec::new();
        for r in rows {
            results.push(self.find_by_id(r.id).await?);
        }
        Ok(results)
    }

    pub async fn list_all(&self, limit: i64, offset: i64) -> Result<Vec<EvalRun>> {
        let rows: Vec<(uuid::Uuid,)> =
            sqlx::query_as("SELECT id FROM eval_runs ORDER BY created_at DESC LIMIT $1 OFFSET $2")
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
                .map_err(db_err)?;

        let mut results = Vec::new();
        for (id,) in rows {
            results.push(self.find_by_id(id).await?);
        }
        Ok(results)
    }

    pub async fn save_error(&self, id: Uuid, message: &str) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE eval_runs
            SET status = 'error'::eval_run_status, error_message = $2,
                completed_at = NOW(), updated_at = NOW()
            WHERE id = $1
            "#,
            id,
            message,
        )
        .execute(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(())
    }

    /// Cancel a run that is still pending/running, or hard-delete a completed/errored run.
    /// Returns `true` if a row was affected, `false` if the ID did not exist.
    pub async fn cancel_or_delete(&self, id: Uuid) -> Result<bool> {
        // If the run is still active, transition it to cancelled first so any
        // in-flight background tasks can observe the status change.
        let result = sqlx::query(
            r#"
            UPDATE eval_runs
            SET status = CASE
                WHEN status IN ('pending'::eval_run_status, 'running'::eval_run_status)
                THEN 'cancelled'::eval_run_status
                ELSE status
            END,
            updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(result.rows_affected() > 0)
    }
}

fn parse_status(s: &str) -> EvalRunStatus {
    match s {
        "running" => EvalRunStatus::Running,
        "complete" => EvalRunStatus::Complete,
        "error" => EvalRunStatus::Error,
        "cancelled" => EvalRunStatus::Cancelled,
        _ => EvalRunStatus::Pending,
    }
}
