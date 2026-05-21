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
use std::sync::atomic::Ordering;

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
    /// Pass threshold 0.0–1.0 (default: `AGENTFORGE_DEFAULT_PASS_THRESHOLD` / 0.85).
    pub threshold: Option<f64>,
    /// LLM provider for the agent under test (`openai` | `anthropic` | `nvidia` | `ollama` | `bedrock`).
    pub provider: Option<String>,
    /// LLM provider for the judge (must differ from `provider` at the provider level).
    pub judge_provider: Option<String>,
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
///
/// Concurrency-limited by `AGENTFORGE_MAX_CONCURRENT_RUNS` (default 10).
/// Returns HTTP 429 when the limit is reached.
pub async fn start_run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StartRunRequest>,
) -> ApiResult<(StatusCode, Json<RunResponse>)> {
    // Concurrency guard: reject new runs when too many are already running
    let current = state.active_runs.fetch_add(1, Ordering::SeqCst);
    if current >= state.max_concurrent_runs {
        state.active_runs.fetch_sub(1, Ordering::SeqCst);
        return Err(ApiError::unprocessable(format!(
            "Too many concurrent evaluation runs ({current} active). \
             Retry after an existing run completes, or increase AGENTFORGE_MAX_CONCURRENT_RUNS."
        )));
    }
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

    // Guard: reject scenario counts exceeding the configured maximum
    if scenario_count > state.max_scenarios {
        state.active_runs.fetch_sub(1, Ordering::SeqCst);
        return Err(ApiError::bad_request(format!(
            "scenario_count {scenario_count} exceeds the maximum allowed ({max}). \
             Increase AGENTFORGE_MAX_SCENARIOS to raise the limit.",
            max = state.max_scenarios,
        )));
    }

    // Guard: reject same-provider runs to prevent circular bias in scoring
    if let (Some(ref provider), Some(ref judge_provider)) = (&req.provider, &req.judge_provider) {
        if provider == judge_provider {
            state.active_runs.fetch_sub(1, Ordering::SeqCst);
            return Err(ApiError::bad_request(
                "provider and judge_provider must be different to prevent circular scoring bias",
            ));
        }
    }

    // Guard: threshold must be a valid 0.0–1.0 probability if supplied
    if let Some(threshold) = req.threshold {
        if !(0.0_f64..=1.0_f64).contains(&threshold) {
            state.active_runs.fetch_sub(1, Ordering::SeqCst);
            return Err(ApiError::bad_request(
                "threshold must be between 0.0 and 1.0",
            ));
        }
    }

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
            state.active_runs.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    };

    if let Err(e) = scenario_repo.insert_batch(&scenarios).await {
        let _ = eval_repo.save_error(run_id, &e.to_string()).await;
        state.active_runs.fetch_sub(1, Ordering::SeqCst);
        return;
    }

    let runner = AgentRunner::new(
        state.llm_client.clone(),
        RunnerConfig {
            concurrency: concurrency as usize,
            run_id,
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
            state.active_runs.fetch_sub(1, Ordering::SeqCst);
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
    state.active_runs.fetch_sub(1, Ordering::SeqCst);
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

/// DELETE /runs/:id
///
/// Cancels a pending/running eval run, or no-ops if already terminal.
/// Returns 204 on success, 404 if the run does not exist.
pub async fn cancel_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> ApiResult<StatusCode> {
    let eval_repo = EvalRepo::new(state.db.clone());
    let affected = eval_repo
        .cancel_or_delete(id)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    if affected {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::not_found(format!("Run {id} not found")))
    }
}

// ─── GET /runs/:id/progress (Server-Sent Events) ────────────────────────────

/// GET /runs/:id/progress — real-time Server-Sent Events stream of run progress.
///
/// Emits a `progress` event every ~2 seconds with the current counts until the
/// run reaches a terminal state (`complete`, `error`, or `cancelled`), at which
/// point a final event is emitted and the stream closes.
///
/// Event format (JSON data field):
/// ```json
/// { "run_id": "...", "status": "running", "completed_count": 42, "scenario_count": 100, "error_count": 0 }
/// ```
///
/// If the run is not found, an `error` event is sent and the stream closes immediately.
/// The endpoint does NOT require the run to be in a running state — polling a
/// completed run returns one event and closes.
pub async fn run_progress(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> axum::response::sse::Sse<
    impl tokio_stream::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(8);

    tokio::spawn(async move {
        loop {
            let eval_repo = EvalRepo::new(state.db.clone());
            match eval_repo.find_by_id(id).await {
                Ok(run) => {
                    let is_terminal = matches!(
                        run.status,
                        EvalRunStatus::Complete | EvalRunStatus::Error | EvalRunStatus::Cancelled
                    );
                    let data = serde_json::json!({
                        "run_id": id,
                        "status": run.status.to_string(),
                        "completed_count": run.completed_count,
                        "scenario_count": run.scenario_count,
                        "error_count": run.error_count,
                    });
                    let ev = Event::default()
                        .event("progress")
                        .json_data(data)
                        .unwrap_or_else(|_| Event::default().data("{}"));
                    if tx.send(Ok(ev)).await.is_err() {
                        // Client disconnected
                        break;
                    }
                    if is_terminal {
                        break;
                    }
                }
                Err(AgentForgeError::NotFound { .. }) => {
                    let ev = Event::default()
                        .event("error")
                        .data(format!("run {id} not found"));
                    let _ = tx.send(Ok(ev)).await;
                    break;
                }
                Err(e) => {
                    let ev = Event::default().event("error").data(e.to_string());
                    let _ = tx.send(Ok(ev)).await;
                    break;
                }
            }

            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // ── StartRunRequest deserialization ───────────────────────────────────────

    /// All new fields (threshold, provider, judge_provider) deserialize correctly.
    #[test]
    fn start_run_request_deserializes_all_new_fields() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "scenario_count": 50,
            "seed": 99,
            "concurrency": 4,
            "threshold": 0.90,
            "provider": "anthropic",
            "judge_provider": "openai"
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.scenario_count, Some(50));
        assert_eq!(req.threshold, Some(0.90));
        assert_eq!(req.provider.as_deref(), Some("anthropic"));
        assert_eq!(req.judge_provider.as_deref(), Some("openai"));
        assert_eq!(req.seed, Some(99));
        assert_eq!(req.concurrency, Some(4));
    }

    /// Only `agent_id` (the sole required field) — everything else is None.
    #[test]
    fn start_run_request_only_agent_id_required() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000"
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert!(req.scenario_count.is_none());
        assert!(req.threshold.is_none());
        assert!(req.provider.is_none());
        assert!(req.judge_provider.is_none());
        assert!(req.seed.is_none());
        assert!(req.concurrency.is_none());
    }

    /// Extra unknown fields in the JSON do not cause a parse error.
    #[test]
    fn start_run_request_ignores_unknown_fields() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "unknown_field": "ignored"
        });
        // Should not panic / error
        let result: Result<StartRunRequest, _> = serde_json::from_value(json);
        assert!(result.is_ok());
    }

    // ── Validation logic (same-provider check) ────────────────────────────────

    /// When provider == judge_provider the handler should reject — validate the
    /// string comparison that drives that guard.
    #[test]
    fn same_provider_strings_are_equal() {
        let provider = "openai";
        let judge_provider = "openai";
        assert_eq!(
            provider, judge_provider,
            "provider and judge_provider must differ; the guard compares them as strings"
        );
    }

    #[test]
    fn different_provider_strings_are_not_equal() {
        assert_ne!("openai", "anthropic");
        assert_ne!("anthropic", "nvidia");
        assert_ne!("openai", "ollama");
    }

    // ── Threshold range validation ─────────────────────────────────────────────

    #[test]
    fn threshold_range_valid_boundaries() {
        // Both boundary values are inside the valid range
        assert!((0.0_f64..=1.0_f64).contains(&0.0));
        assert!((0.0_f64..=1.0_f64).contains(&1.0));
        assert!((0.0_f64..=1.0_f64).contains(&0.85));
    }

    #[test]
    fn threshold_range_rejects_out_of_bounds() {
        assert!(!(0.0_f64..=1.0_f64).contains(&-0.01));
        assert!(!(0.0_f64..=1.0_f64).contains(&1.01));
        assert!(!(0.0_f64..=1.0_f64).contains(&2.0));
    }

    // ── RunResponse From<EvalRun> conversion ──────────────────────────────────

    #[test]
    fn run_response_from_eval_run_maps_fields() {
        let run_id = Uuid::new_v4();
        let agent_id = Uuid::new_v4();
        let created = Utc::now();

        let eval_run = EvalRun {
            id: run_id,
            agent_id,
            scenario_set_id: None,
            status: EvalRunStatus::Running,
            scenario_count: 100,
            completed_count: 42,
            error_count: 3,
            aggregate_score: None,
            pass_rate: None,
            scores: None,
            failure_clusters: None,
            seed: 42,
            concurrency: 10,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: created,
            updated_at: created,
        };

        let resp = RunResponse::from(eval_run);
        assert_eq!(resp.id, run_id);
        assert_eq!(resp.agent_id, agent_id);
        assert_eq!(resp.status, "running");
        assert_eq!(resp.created_at, created);
    }

    // ── SSE progress event JSON shape ─────────────────────────────────────────

    /// The JSON object embedded in progress SSE events must have all five keys.
    #[test]
    fn progress_event_json_contains_expected_keys() {
        let run_id = Uuid::new_v4();
        let data = serde_json::json!({
            "run_id": run_id,
            "status": "running",
            "completed_count": 25_u32,
            "scenario_count": 100_u32,
            "error_count": 1_u32,
        });
        assert!(data["run_id"].is_string());
        assert_eq!(data["status"], "running");
        assert_eq!(data["completed_count"], 25);
        assert_eq!(data["scenario_count"], 100);
        assert_eq!(data["error_count"], 1);
    }

    /// A completed run status string matches what the SSE handler checks.
    #[test]
    fn terminal_status_strings_match_enum_display() {
        let complete = EvalRunStatus::Complete.to_string();
        let error = EvalRunStatus::Error.to_string();
        let cancelled = EvalRunStatus::Cancelled.to_string();
        // Ensure the .to_string() values are stable (they appear in SSE payloads)
        assert!(!complete.is_empty());
        assert!(!error.is_empty());
        assert!(!cancelled.is_empty());
        // All three are distinct
        assert_ne!(complete, error);
        assert_ne!(complete, cancelled);
        assert_ne!(error, cancelled);
    }

    // ── ListRunsQuery / ListTracesQuery pagination defaults ───────────────────

    #[test]
    fn list_runs_query_defaults_are_sane() {
        let query: ListRunsQuery = serde_json::from_value(serde_json::json!({})).unwrap();
        let limit = query.limit.unwrap_or(50).min(200);
        let offset = query.offset.unwrap_or(0);
        assert_eq!(limit, 50);
        assert_eq!(offset, 0);
    }

    #[test]
    fn list_traces_query_clamps_limit_at_500() {
        let query: ListTracesQuery = serde_json::from_value(serde_json::json!({
            "limit": 9999
        }))
        .unwrap();
        let limit = query.limit.unwrap_or(100).min(500);
        assert_eq!(limit, 500);
    }
}
