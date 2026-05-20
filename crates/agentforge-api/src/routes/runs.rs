use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use agentforge_core::{AgentForgeError, EvalRun, EvalRunStatus, Trace};
use agentforge_db::{
    agent_repo::AgentRepo, eval_repo::EvalRepo, scenario_repo::ScenarioRepo, trace_repo::TraceRepo,
};
use agentforge_runner::{AgentRunner, RunnerConfig};
use agentforge_scenarios::ScenarioGeneratorConfig;
use agentforge_scorer::score_run;

use crate::{
    error::{ApiError, ApiResult},
    state::AppState,
};

#[derive(Debug, Deserialize)]
pub struct StartRunRequest {
    pub agent_id: Uuid,
    pub scenario_count: Option<u32>,
    pub seed: Option<i64>,
    pub concurrency: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ListRunsQuery {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct ListTracesQuery {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct RunResponse {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl From<EvalRun> for RunResponse {
    fn from(r: EvalRun) -> Self {
        Self {
            id: r.id,
            agent_id: r.agent_id,
            status: r.status.to_string(),
            created_at: r.created_at,
        }
    }
}

/// GET /runs — list all eval runs (paginated)
pub async fn list_runs(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListRunsQuery>,
) -> ApiResult<Json<Vec<RunResponse>>> {
    let eval_repo = EvalRepo::new(state.db.clone());
    let limit = params.limit.unwrap_or(50).min(200);
    let offset = params.offset.unwrap_or(0);
    let runs = eval_repo
        .list_all(limit, offset)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(runs.into_iter().map(Into::into).collect()))
}

/// POST /runs — start a new evaluation run (returns 202 immediately, runs in background)
pub async fn start_run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StartRunRequest>,
) -> ApiResult<(StatusCode, Json<RunResponse>)> {
    let agent_repo = AgentRepo::new(state.db.clone());
    let agent_version = agent_repo
        .find_by_id(req.agent_id)
        .await
        .map_err(|e| match e {
            AgentForgeError::NotFound { .. } => {
                ApiError::not_found(format!("Agent {} not found", req.agent_id))
            }
            other => ApiError::internal(other.to_string()),
        })?;

    let agent_file: agentforge_core::AgentFile = agent_version.file_content.clone();

    let scenario_count = req
        .scenario_count
        .or_else(|| {
            agent_version
                .file_content
                .eval_hints
                .as_ref()
                .and_then(|h| h.scenario_count)
        })
        .unwrap_or(100);
    let concurrency = req.concurrency.unwrap_or(10);
    let seed = req.seed.unwrap_or(42);

    let new_run = EvalRun {
        id: Uuid::new_v4(),
        agent_id: req.agent_id,
        scenario_set_id: None,
        status: EvalRunStatus::Pending,
        scenario_count,
        completed_count: 0,
        error_count: 0,
        aggregate_score: None,
        pass_rate: None,
        scores: None,
        failure_clusters: None,
        seed: seed as u32,
        concurrency,
        error_message: None,
        started_at: None,
        completed_at: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    let eval_repo = EvalRepo::new(state.db.clone());
    let run = eval_repo
        .insert(&new_run)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let run_id = run.id;

    let state_clone = state.clone();
    tokio::spawn(async move {
        run_evaluation_background(
            state_clone,
            run_id,
            agent_file,
            req.agent_id,
            scenario_count,
            concurrency,
        )
        .await;
    });

    Ok((StatusCode::ACCEPTED, Json(run.into())))
}

async fn run_evaluation_background(
    state: Arc<AppState>,
    run_id: Uuid,
    agent: agentforge_core::AgentFile,
    agent_id: Uuid,
    scenario_count: u32,
    concurrency: u32,
) {
    let eval_repo = EvalRepo::new(state.db.clone());
    let scenario_repo = ScenarioRepo::new(state.db.clone());
    let trace_repo = TraceRepo::new(state.db.clone());

    let _ = eval_repo
        .update_status(run_id, &EvalRunStatus::Running)
        .await;

    let scenarios = match agentforge_scenarios::generate_scenarios(
        &agent,
        &ScenarioGeneratorConfig {
            total_count: scenario_count,
            agent_id,
            llm_base_url: Some(state.scorer_config.judge_base_url.clone()),
            llm_api_key: if state.scorer_config.judge_api_key.is_empty() {
                None
            } else {
                Some(state.scorer_config.judge_api_key.clone())
            },
            llm_model: Some(state.scorer_config.judge_model.clone()),
            ..Default::default()
        },
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            let _ = eval_repo.save_error(run_id, &e.to_string()).await;
            return;
        }
    };

    if let Err(e) = scenario_repo.insert_batch(&scenarios).await {
        let _ = eval_repo.save_error(run_id, &e.to_string()).await;
        return;
    }

    let runner = AgentRunner::new(
        state.llm_client.clone(),
        RunnerConfig {
            concurrency: concurrency as usize,
            ..Default::default()
        },
    );
    let mut traces = runner.run(&agent, scenarios.clone(), None).await.traces;

    let scorecard = match score_run(
        &mut traces,
        &scenarios,
        &agent,
        run_id,
        &state.scorer_config,
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            let _ = eval_repo.save_error(run_id, &e.to_string()).await;
            return;
        }
    };

    for trace in &traces {
        let _ = trace_repo.insert(trace).await;
    }

    let _ = eval_repo
        .save_scores(
            run_id,
            &scorecard.dimension_scores,
            scorecard.aggregate_score,
            scorecard.pass_rate,
            &scorecard.failure_clusters,
        )
        .await;

    // Update completed/error counts now that we know the final tally
    let completed = scorecard.total_scenarios.saturating_sub(scorecard.errors);
    let _ = eval_repo
        .update_progress(run_id, completed, scorecard.errors)
        .await;

    let _ = eval_repo
        .update_status(run_id, &EvalRunStatus::Complete)
        .await;
    tracing::info!(%run_id, aggregate = scorecard.aggregate_score, passed = scorecard.passed, errors = scorecard.errors, "Evaluation complete");
}

/// GET /runs/:id
pub async fn get_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<RunResponse>> {
    let eval_repo = EvalRepo::new(state.db.clone());
    let run = eval_repo.find_by_id(id).await.map_err(|e| match e {
        AgentForgeError::NotFound { .. } => ApiError::not_found(format!("Run {id} not found")),
        other => ApiError::internal(other.to_string()),
    })?;
    Ok(Json(run.into()))
}

/// GET /runs/:id/scorecard
pub async fn get_scorecard(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<EvalRun>> {
    let eval_repo = EvalRepo::new(state.db.clone());
    let run = eval_repo.find_by_id(id).await.map_err(|e| match e {
        AgentForgeError::NotFound { .. } => ApiError::not_found(format!("Run {id} not found")),
        other => ApiError::internal(other.to_string()),
    })?;
    Ok(Json(run))
}

/// GET /runs/:id/traces — list traces for a run (paginated; default limit=100, max=500)
pub async fn list_traces(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(params): Query<ListTracesQuery>,
) -> ApiResult<Json<Vec<Trace>>> {
    let trace_repo = TraceRepo::new(state.db.clone());
    let limit = params.limit.unwrap_or(100).min(500);
    let offset = params.offset.unwrap_or(0);
    let traces = trace_repo
        .list_by_run_paginated(id, limit, offset)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(traces))
}
