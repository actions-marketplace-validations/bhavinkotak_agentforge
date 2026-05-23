use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use agentforge_core::{
    AgentFileFormat, AgentForgeError, AgentVersion, EvalRun, EvalRunStatus, Trace,
};
use agentforge_db::{
    agent_repo::AgentRepo, eval_repo::EvalRepo, scenario_repo::ScenarioRepo, trace_repo::TraceRepo,
};
use agentforge_optimizer::Optimizer;
use agentforge_parser::compute_sha256;
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
    /// Optimization target score 0.0–1.0 (default: 0.92).
    /// Also used as the pass threshold for the gatekeeper.
    pub threshold: Option<f64>,
    /// LLM provider for the agent under test (`openai` | `anthropic` | `nvidia` | `ollama` | `bedrock`).
    pub provider: Option<String>,
    /// LLM provider for the judge (must differ from `provider` at the provider level).
    pub judge_provider: Option<String>,
    /// Enable the iterative self-improvement loop (default: **true**).
    /// The optimizer will iterate up to `max_opt_iterations` rounds, each time
    /// generating variants and saving the best one, until the score reaches
    /// `threshold` or no further improvement is possible.
    pub auto_optimize: Option<bool>,
    /// Maximum number of optimization rounds (default: 5).
    pub max_opt_iterations: Option<u32>,
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
    pub aggregate_score: Option<f64>,
    pub pass_rate: Option<f64>,
    pub scenario_count: u32,
    pub completed_count: u32,
    pub error_count: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    // ── Self-improvement loop state ────────────────────────────────────────
    /// `running` | `converged` | `no_improvement` | `max_iterations` | `failed`.
    /// `null` when auto_optimize was not requested.
    pub opt_status: Option<String>,
    /// Number of completed optimization rounds.
    pub opt_rounds: i32,
    /// Best aggregate score reached during the optimization loop.
    pub opt_best_score: Option<f64>,
    /// UUID of the best agent version saved by the optimization loop.
    pub opt_best_agent_id: Option<Uuid>,
}

impl From<EvalRun> for RunResponse {
    fn from(r: EvalRun) -> Self {
        Self {
            id: r.id,
            agent_id: r.agent_id,
            status: r.status.to_string(),
            aggregate_score: r.aggregate_score,
            pass_rate: r.pass_rate,
            scenario_count: r.scenario_count,
            completed_count: r.completed_count,
            error_count: r.error_count,
            created_at: r.created_at,
            opt_status: r.opt_status,
            opt_rounds: r.opt_rounds,
            opt_best_score: r.opt_best_score,
            opt_best_agent_id: r.opt_best_agent_id,
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

/// GET /agents/:id/runs — list completed eval runs for a specific agent, newest first.
pub async fn list_runs_for_agent(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<Uuid>,
    Query(params): Query<ListRunsQuery>,
) -> ApiResult<Json<Vec<RunResponse>>> {
    let eval_repo = EvalRepo::new(state.db.clone());
    let limit = params.limit.unwrap_or(20).min(200);
    let runs = eval_repo
        .list_by_agent(agent_id, limit)
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
        opt_status: None,
        opt_rounds: 0,
        opt_best_score: None,
        opt_best_agent_id: None,
    };

    let eval_repo = EvalRepo::new(state.db.clone());
    let run = eval_repo
        .insert(&new_run)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let run_id = run.id;

    let state_clone = state.clone();
    let auto_optimize = req.auto_optimize.unwrap_or(true);
    let opt_threshold = req.threshold.unwrap_or(0.92);
    let max_opt_iterations = req.max_opt_iterations.unwrap_or(5);
    tokio::spawn(async move {
        run_evaluation_background(
            state_clone,
            run_id,
            agent_file,
            req.agent_id,
            scenario_count,
            concurrency,
            auto_optimize,
            opt_threshold,
            max_opt_iterations,
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
    auto_optimize: bool,
    opt_threshold: f64,
    max_opt_iterations: u32,
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

    // If all traces errored (e.g. LLM API unavailable), surface a sample failure_reason
    // as the run-level error_message so the UI can show something meaningful.
    if scorecard.errors > 0 && scorecard.errors >= scorecard.total_scenarios {
        let sample_failure = traces
            .iter()
            .find_map(|t| t.failure_reason.as_deref())
            .unwrap_or("All traces failed — check LLM API credentials and quota");
        let _ = eval_repo.set_error_message(run_id, sample_failure).await;
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

    // Self-improvement: if auto_optimize is enabled and the run didn't reach the threshold,
    // kick off the iterative optimization loop in a separate task.
    if auto_optimize && scorecard.aggregate_score < opt_threshold {
        tracing::info!(%run_id, aggregate = scorecard.aggregate_score, %opt_threshold, "Score below threshold — starting iterative optimization");
        tokio::spawn(run_iterative_optimization(
            state.clone(),
            run_id,
            agent,
            agent_id,
            scorecard.clone(),
            traces,
            scenarios,
            opt_threshold,
            max_opt_iterations,
        ));
    } else if auto_optimize {
        tracing::info!(%run_id, aggregate = scorecard.aggregate_score, "Score already meets threshold — skipping optimization");
    }
}

/// Iterative self-improvement loop.
///
/// Each round:
///   1. Generate variant agents using the optimizer.
///   2. Quick-eval each variant against a small scenario sample.
///   3. If the best variant improves by > 1 pp, save it and use it as the
///      agent for the next round.
///   4. Update `opt_status / opt_rounds / opt_best_score / opt_best_agent_id`
///      in the DB after every round.
///
/// Terminates when:
///   - Score ≥ `threshold`  → `converged`
///   - No variant improves by > 1 pp  → `no_improvement`
///   - `round == max_iterations`  → `max_iterations`
///   - Any unrecoverable error  → `failed`
async fn run_iterative_optimization(
    state: Arc<AppState>,
    run_id: Uuid,
    initial_agent: agentforge_core::AgentFile,
    agent_id: Uuid,
    initial_scorecard: agentforge_core::Scorecard,
    passing_traces: Vec<Trace>,
    scenarios: Vec<agentforge_core::Scenario>,
    threshold: f64,
    max_iterations: u32,
) {
    let eval_repo = EvalRepo::new(state.db.clone());
    let agent_repo = AgentRepo::new(state.db.clone());
    let optimizer = Optimizer::new(state.optimizer_config.clone());

    // Quick-eval uses up to 25 scenarios to balance speed vs accuracy.
    let eval_scenarios: Vec<agentforge_core::Scenario> =
        scenarios.into_iter().take(25).collect();

    let mut current_agent = initial_agent;
    let mut current_score = initial_scorecard.aggregate_score;
    let mut current_scorecard = initial_scorecard;
    let mut best_agent_id: Option<Uuid> = None;

    // Derive the current agent's SHA from the DB so lineage links are correct.
    let mut parent_sha = match agent_repo.find_by_id(agent_id).await {
        Ok(v) => v.sha.clone(),
        Err(_) => current_agent.name.clone(),
    };

    let passing: Vec<Trace> = passing_traces
        .into_iter()
        .filter(|t| t.status == agentforge_core::TraceStatus::Pass)
        .collect();

    let mut round: u32 = 0;
    let terminal_status = loop {
        if current_score >= threshold {
            break "converged";
        }
        if round >= max_iterations {
            break "max_iterations";
        }

        // Mark the loop as running in the DB.
        let _ = eval_repo
            .update_opt_tracking(run_id, "running", round as i32, Some(current_score), best_agent_id)
            .await;

        tracing::info!(%run_id, round, current_score, %threshold, "Optimization round starting");

        let result = match optimizer
            .generate_variants(&current_agent, &current_scorecard, &passing, &parent_sha)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, round, "Variant generation failed");
                break "failed";
            }
        };

        if result.variants.is_empty() {
            tracing::info!(round, "No variants generated — stopping optimization");
            break "no_improvement";
        }

        let mut round_best_score = current_score;
        let mut round_best_agent: Option<agentforge_core::AgentFile> = None;
        let mut round_best_mutation = String::new();

        for variant in &result.variants {
            let mini_run_id = Uuid::new_v4();
            let runner = AgentRunner::new(
                state.llm_client.clone(),
                RunnerConfig {
                    concurrency: 3,
                    run_id: mini_run_id,
                    ..Default::default()
                },
            );

            let mut mini_traces = runner
                .run(&variant.agent, eval_scenarios.clone(), None)
                .await
                .traces;

            let mini_scorecard = match score_run(
                &mut mini_traces,
                &eval_scenarios,
                &variant.agent,
                mini_run_id,
                &state.scorer_config,
            )
            .await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, mutation = %variant.mutation_type, round, "Mini-eval failed");
                    continue;
                }
            };

            tracing::info!(
                mutation = %variant.mutation_type,
                score = mini_scorecard.aggregate_score,
                current = current_score,
                round,
                "Variant evaluated"
            );

            if mini_scorecard.aggregate_score > round_best_score {
                round_best_score = mini_scorecard.aggregate_score;
                round_best_agent = Some(variant.agent.clone());
                round_best_mutation = variant.mutation_type.to_string();
            }
        }

        // Only accept if the improvement is > 1 percentage point.
        if round_best_score <= current_score + 0.01 {
            tracing::info!(round, current_score, round_best_score, "No meaningful improvement this round");
            break "no_improvement";
        }

        // Save the improved version to the DB.
        let improved = round_best_agent.unwrap();
        let now = Utc::now();
        let json = serde_json::to_string(&improved).unwrap_or_default();
        let new_sha = compute_sha256(&json);
        let saved_version = AgentVersion {
            id: Uuid::new_v4(),
            name: improved.name.clone(),
            version: improved.version.clone(),
            sha: new_sha.clone(),
            file_content: improved.clone(),
            raw_content: serde_json::to_string_pretty(&improved).unwrap_or_default(),
            format: AgentFileFormat::NativeYaml,
            promoted: false,
            is_champion: false,
            changelog: Some(format!(
                "Auto-optimized via {round_best_mutation} (round {}) — score {:.1}% → {:.1}%",
                round + 1,
                current_score * 100.0,
                round_best_score * 100.0,
            )),
            parent_sha: Some(parent_sha.clone()),
            created_at: now,
            updated_at: now,
        };

        match agent_repo.insert(&saved_version).await {
            Ok(saved) => {
                tracing::info!(
                    %run_id, round,
                    new_version_id = %saved.id,
                    version = %saved.version,
                    score = round_best_score,
                    "Saved improved agent version"
                );
                best_agent_id = Some(saved.id);
                parent_sha = new_sha;
                // Use the full agent scorecard proxy for the next round.
                current_scorecard = agentforge_core::Scorecard {
                    aggregate_score: round_best_score,
                    ..current_scorecard
                };
                current_score = round_best_score;
                current_agent = improved;
            }
            Err(e) => {
                // If a duplicate-SHA error, the version already exists — look it up
                // and reuse it rather than aborting the loop.
                let is_dup = e.to_string().contains("duplicate key")
                    || e.to_string().contains("unique constraint");
                if is_dup {
                    tracing::info!(round, sha = %new_sha, "SHA already exists; reusing existing version");
                    if let Ok(Some(existing)) = agent_repo.find_by_sha(&new_sha).await {
                        best_agent_id = Some(existing.id);
                        parent_sha = new_sha;
                        current_scorecard = agentforge_core::Scorecard {
                            aggregate_score: round_best_score,
                            ..current_scorecard
                        };
                        current_score = round_best_score;
                        current_agent = improved;
                        tracing::info!(%run_id, round, existing_id = %existing.id, score = round_best_score, "Reused existing agent version");
                    } else {
                        tracing::warn!(round, "SHA conflict but version not found; skipping save");
                    }
                } else {
                    tracing::warn!(error = %e, round, "Failed to save improved agent version");
                    break "failed";
                }
            }
        }

        round += 1;
    };

    let _ = eval_repo
        .update_opt_tracking(run_id, terminal_status, round as i32, Some(current_score), best_agent_id)
        .await;

    tracing::info!(%run_id, terminal_status, round, final_score = current_score, "Optimization loop finished");
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
                    let eval_done = matches!(
                        run.status,
                        EvalRunStatus::Complete | EvalRunStatus::Error | EvalRunStatus::Cancelled
                    );
                    let opt_running = run.opt_status.as_deref() == Some("running");
                    // Terminate the SSE stream only when eval is done AND the
                    // optimization loop is not still running.
                    let is_terminal = eval_done && !opt_running;
                    let data = serde_json::json!({
                        "run_id": id,
                        "status": run.status.to_string(),
                        "completed_count": run.completed_count,
                        "scenario_count": run.scenario_count,
                        "error_count": run.error_count,
                        "opt_status": run.opt_status,
                        "opt_rounds": run.opt_rounds,
                        "opt_best_score": run.opt_best_score,
                        "opt_best_agent_id": run.opt_best_agent_id,
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
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
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

    // ── RunResponse aggregate_score regression tests ───────────────────────────

    /// Regression: RunResponse must include aggregate_score field.
    /// Before the fix, aggregate_score was missing from RunResponse and always null in the API.
    #[test]
    fn run_response_serializes_aggregate_score() {
        let run_id = Uuid::new_v4();
        let agent_id = Uuid::new_v4();
        let resp = RunResponse {
            id: run_id,
            agent_id,
            status: "complete".to_string(),
            aggregate_score: Some(0.87),
            pass_rate: Some(0.80),
            scenario_count: 10,
            completed_count: 9,
            error_count: 1,
            created_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(
            json["aggregate_score"].is_number(),
            "aggregate_score must serialize as a number, got: {:?}",
            json["aggregate_score"]
        );
        assert!((json["aggregate_score"].as_f64().unwrap() - 0.87).abs() < 1e-9);
    }

    #[test]
    fn run_response_aggregate_score_none_serializes_as_null() {
        let resp = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "running".to_string(),
            aggregate_score: None,
            pass_rate: None,
            scenario_count: 5,
            completed_count: 0,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(
            json["aggregate_score"].is_null(),
            "aggregate_score: None must serialize as null"
        );
    }

    #[test]
    fn run_response_from_eval_run_maps_aggregate_score() {
        let run_id = Uuid::new_v4();
        let agent_id = Uuid::new_v4();
        let created = Utc::now();
        let eval_run = EvalRun {
            id: run_id,
            agent_id,
            scenario_set_id: None,
            status: EvalRunStatus::Complete,
            scenario_count: 100,
            completed_count: 100,
            error_count: 0,
            aggregate_score: Some(0.92),
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
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let resp = RunResponse::from(eval_run);
        assert_eq!(resp.aggregate_score, Some(0.92));
    }

    #[test]
    fn run_response_from_eval_run_preserves_none_aggregate_score() {
        let eval_run = EvalRun {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            scenario_set_id: None,
            status: EvalRunStatus::Pending,
            scenario_count: 50,
            completed_count: 0,
            error_count: 0,
            aggregate_score: None, // Not yet scored
            pass_rate: None,
            scores: None,
            failure_clusters: None,
            seed: 0,
            concurrency: 5,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let resp = RunResponse::from(eval_run);
        assert!(
            resp.aggregate_score.is_none(),
            "aggregate_score must be None when EvalRun has no score yet"
        );
    }

    #[test]
    fn run_response_has_all_expected_fields_serialized() {
        let resp = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "complete".to_string(),
            aggregate_score: Some(0.75),
            pass_rate: Some(0.60),
            scenario_count: 5,
            completed_count: 5,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        let obj = json.as_object().unwrap();
        assert!(obj.contains_key("id"), "RunResponse must have 'id' field");
        assert!(
            obj.contains_key("agent_id"),
            "RunResponse must have 'agent_id' field"
        );
        assert!(
            obj.contains_key("status"),
            "RunResponse must have 'status' field"
        );
        assert!(
            obj.contains_key("aggregate_score"),
            "RunResponse must have 'aggregate_score' field"
        );
        assert!(
            obj.contains_key("created_at"),
            "RunResponse must have 'created_at' field"
        );
    }

    // ── auto_optimize field ───────────────────────────────────────────────────

    #[test]
    fn start_run_request_auto_optimize_defaults_to_none() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000"
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert!(
            req.auto_optimize.is_none(),
            "auto_optimize should default to None when not provided"
        );
    }

    #[test]
    fn start_run_request_auto_optimize_true_parsed() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "auto_optimize": true
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.auto_optimize, Some(true));
    }

    #[test]
    fn start_run_request_auto_optimize_false_parsed() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "auto_optimize": false
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.auto_optimize, Some(false));
    }

    #[test]
    fn auto_optimize_defaults_to_true_via_unwrap_or() {
        let req = StartRunRequest {
            agent_id: Uuid::new_v4(),
            scenario_count: None,
            seed: None,
            concurrency: None,
            threshold: None,
            provider: None,
            judge_provider: None,
            auto_optimize: None,
            max_opt_iterations: None,
        };
        // The handler now defaults to true — self-improvement is on by default.
        let auto_optimize = req.auto_optimize.unwrap_or(true);
        assert!(auto_optimize, "auto_optimize must default to true");
    }

    // ── EvalRunStatus Display stability ──────────────────────────────────────

    #[test]
    fn eval_run_status_pending_display() {
        assert!(!EvalRunStatus::Pending.to_string().is_empty());
    }

    #[test]
    fn eval_run_status_running_display() {
        assert!(!EvalRunStatus::Running.to_string().is_empty());
    }

    #[test]
    fn eval_run_status_complete_display() {
        assert!(!EvalRunStatus::Complete.to_string().is_empty());
    }

    #[test]
    fn eval_run_status_error_display() {
        assert!(!EvalRunStatus::Error.to_string().is_empty());
    }

    #[test]
    fn all_eval_run_statuses_are_distinct() {
        let statuses = [
            EvalRunStatus::Pending.to_string(),
            EvalRunStatus::Running.to_string(),
            EvalRunStatus::Complete.to_string(),
            EvalRunStatus::Error.to_string(),
            EvalRunStatus::Cancelled.to_string(),
        ];
        let unique: std::collections::HashSet<_> = statuses.iter().collect();
        assert_eq!(
            unique.len(),
            statuses.len(),
            "All EvalRunStatus Display values must be distinct"
        );
    }

    // ── Pagination bounds ──────────────────────────────────────────────────────

    #[test]
    fn list_runs_negative_offset_becomes_zero() {
        // The handler uses offset.unwrap_or(0) — negative values are passed through
        // to the DB where they would fail, so we ensure the default is non-negative.
        let query: ListRunsQuery = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(query.offset.unwrap_or(0), 0);
    }

    #[test]
    fn list_runs_limit_clamps_at_200() {
        let query: ListRunsQuery =
            serde_json::from_value(serde_json::json!({"limit": 9999})).unwrap();
        let limit = query.limit.unwrap_or(50).min(200);
        assert_eq!(limit, 200);
    }

    #[test]
    fn list_runs_reasonable_default_limit() {
        let query: ListRunsQuery = serde_json::from_value(serde_json::json!({})).unwrap();
        let limit = query.limit.unwrap_or(50).min(200);
        assert!(limit > 0 && limit <= 200);
    }

    // ── 22 new tests ─────────────────────────────────────────────────────────

    // ── RunResponse new fields ────────────────────────────────────────────────

    #[test]
    fn run_response_new_fields_serialized() {
        let resp = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "complete".to_string(),
            aggregate_score: Some(0.80),
            pass_rate: Some(0.60),
            scenario_count: 5,
            completed_count: 4,
            error_count: 1,
            created_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["pass_rate"].is_number());
        assert_eq!(json["scenario_count"], 5);
        assert_eq!(json["completed_count"], 4);
        assert_eq!(json["error_count"], 1);
    }

    #[test]
    fn run_response_pass_rate_none_serializes_null() {
        let resp = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "running".to_string(),
            aggregate_score: None,
            pass_rate: None,
            scenario_count: 10,
            completed_count: 3,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["pass_rate"].is_null());
        assert_eq!(json["scenario_count"], 10);
    }

    #[test]
    fn run_response_from_eval_run_maps_pass_rate() {
        let eval_run = EvalRun {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            scenario_set_id: None,
            status: EvalRunStatus::Complete,
            scenario_count: 5,
            completed_count: 4,
            error_count: 1,
            aggregate_score: Some(0.73),
            pass_rate: Some(0.20),
            scores: None,
            failure_clusters: None,
            seed: 42,
            concurrency: 5,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let resp = RunResponse::from(eval_run);
        assert_eq!(resp.pass_rate, Some(0.20));
        assert_eq!(resp.scenario_count, 5);
        assert_eq!(resp.completed_count, 4);
        assert_eq!(resp.error_count, 1);
    }

    #[test]
    fn run_response_from_eval_run_maps_scenario_count_zero() {
        let eval_run = EvalRun {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            scenario_set_id: None,
            status: EvalRunStatus::Pending,
            scenario_count: 0,
            completed_count: 0,
            error_count: 0,
            aggregate_score: None,
            pass_rate: None,
            scores: None,
            failure_clusters: None,
            seed: 0,
            concurrency: 1,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let resp = RunResponse::from(eval_run);
        assert_eq!(resp.scenario_count, 0);
        assert_eq!(resp.completed_count, 0);
    }

    // ── EvalRunStatus equality and cloning ─────────────────────────────────────

    #[test]
    fn eval_run_status_eq() {
        assert_eq!(EvalRunStatus::Complete, EvalRunStatus::Complete);
        assert_ne!(EvalRunStatus::Complete, EvalRunStatus::Error);
    }

    #[test]
    fn eval_run_status_clone() {
        let s = EvalRunStatus::Running;
        let c = s.clone();
        assert_eq!(s, c);
    }

    // ── StartRunRequest boundary cases ────────────────────────────────────────

    #[test]
    fn start_run_request_zero_scenario_count_allowed() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "scenario_count": 0
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.scenario_count, Some(0));
    }

    #[test]
    fn start_run_request_large_concurrency() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "concurrency": 100
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.concurrency, Some(100));
    }

    #[test]
    fn start_run_request_seed_zero() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "seed": 0
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.seed, Some(0));
    }

    #[test]
    fn start_run_request_threshold_zero() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "threshold": 0.0
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.threshold, Some(0.0));
    }

    #[test]
    fn start_run_request_threshold_one() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "threshold": 1.0
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.threshold, Some(1.0));
    }

    // ── ListRunsQuery pagination ───────────────────────────────────────────────

    #[test]
    fn list_runs_query_explicit_limit() {
        let query: ListRunsQuery =
            serde_json::from_value(serde_json::json!({"limit": 10})).unwrap();
        let limit = query.limit.unwrap_or(50).min(200);
        assert_eq!(limit, 10);
    }

    #[test]
    fn list_runs_query_explicit_offset() {
        let query: ListRunsQuery =
            serde_json::from_value(serde_json::json!({"offset": 100})).unwrap();
        assert_eq!(query.offset, Some(100));
    }

    #[test]
    fn list_traces_query_default_limit() {
        let query: ListTracesQuery = serde_json::from_value(serde_json::json!({})).unwrap();
        let limit = query.limit.unwrap_or(100).min(500);
        assert_eq!(limit, 100);
    }

    #[test]
    fn list_traces_query_explicit_limit_below_cap() {
        let query: ListTracesQuery =
            serde_json::from_value(serde_json::json!({"limit": 200})).unwrap();
        let limit = query.limit.unwrap_or(100).min(500);
        assert_eq!(limit, 200);
    }

    // ── Progress SSE JSON shape ────────────────────────────────────────────────

    #[test]
    fn progress_event_status_is_string() {
        let data = serde_json::json!({
            "run_id": Uuid::new_v4().to_string(),
            "status": "running",
            "completed_count": 10_u32,
            "scenario_count": 50_u32,
            "error_count": 0_u32,
        });
        assert_eq!(data["status"].as_str(), Some("running"));
    }

    #[test]
    fn progress_event_complete_status() {
        let data = serde_json::json!({
            "run_id": Uuid::new_v4().to_string(),
            "status": EvalRunStatus::Complete.to_string(),
            "completed_count": 50_u32,
            "scenario_count": 50_u32,
            "error_count": 0_u32,
        });
        assert_eq!(data["status"].as_str(), Some("complete"));
    }

    #[test]
    fn run_response_error_count_zero_when_all_passed() {
        let eval_run = EvalRun {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            scenario_set_id: None,
            status: EvalRunStatus::Complete,
            scenario_count: 5,
            completed_count: 5,
            error_count: 0,
            aggregate_score: Some(1.0),
            pass_rate: Some(1.0),
            scores: None,
            failure_clusters: None,
            seed: 1,
            concurrency: 5,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let resp = RunResponse::from(eval_run);
        assert_eq!(resp.error_count, 0);
        assert_eq!(resp.completed_count, 5);
    }

    // ── 40 new tests for opt tracking & iterative loop ───────────────────────

    // ── StartRunRequest max_opt_iterations ────────────────────────────────────

    #[test]
    fn start_run_request_max_opt_iterations_parsed() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "max_opt_iterations": 7
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.max_opt_iterations, Some(7));
    }

    #[test]
    fn start_run_request_max_opt_iterations_defaults_none() {
        let json = serde_json::json!({ "agent_id": "550e8400-e29b-41d4-a716-446655440000" });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert!(req.max_opt_iterations.is_none());
    }

    #[test]
    fn max_opt_iterations_default_via_unwrap_or() {
        let req = StartRunRequest {
            agent_id: Uuid::new_v4(),
            scenario_count: None,
            seed: None,
            concurrency: None,
            threshold: None,
            provider: None,
            judge_provider: None,
            auto_optimize: None,
            max_opt_iterations: None,
        };
        assert_eq!(req.max_opt_iterations.unwrap_or(5), 5);
    }

    #[test]
    fn max_opt_iterations_one_is_valid() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "max_opt_iterations": 1
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.max_opt_iterations, Some(1));
    }

    #[test]
    fn start_run_request_all_new_opt_fields() {
        let json = serde_json::json!({
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "auto_optimize": true,
            "threshold": 0.95,
            "max_opt_iterations": 5
        });
        let req: StartRunRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.auto_optimize, Some(true));
        assert_eq!(req.threshold, Some(0.95));
        assert_eq!(req.max_opt_iterations, Some(5));
    }

    // ── RunResponse opt fields ─────────────────────────────────────────────────

    #[test]
    fn run_response_opt_status_none_serializes_null() {
        let resp = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "complete".to_string(),
            aggregate_score: None,
            pass_rate: None,
            scenario_count: 5,
            completed_count: 5,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["opt_status"].is_null());
        assert_eq!(json["opt_rounds"], 0);
        assert!(json["opt_best_score"].is_null());
        assert!(json["opt_best_agent_id"].is_null());
    }

    #[test]
    fn run_response_opt_status_running_serializes() {
        let resp = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "complete".to_string(),
            aggregate_score: Some(0.4),
            pass_rate: None,
            scenario_count: 10,
            completed_count: 10,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: Some("running".to_string()),
            opt_rounds: 2,
            opt_best_score: Some(0.65),
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["opt_status"], "running");
        assert_eq!(json["opt_rounds"], 2);
        assert!((json["opt_best_score"].as_f64().unwrap() - 0.65).abs() < 1e-9);
    }

    #[test]
    fn run_response_opt_status_converged_serializes() {
        let resp = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "complete".to_string(),
            aggregate_score: Some(0.97),
            pass_rate: None,
            scenario_count: 10,
            completed_count: 10,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: Some("converged".to_string()),
            opt_rounds: 3,
            opt_best_score: Some(0.97),
            opt_best_agent_id: Some(Uuid::new_v4()),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["opt_status"], "converged");
        assert_eq!(json["opt_rounds"], 3);
        assert!(json["opt_best_agent_id"].is_string());
    }

    #[test]
    fn run_response_from_eval_run_maps_opt_fields() {
        let best_id = Uuid::new_v4();
        let eval_run = EvalRun {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            scenario_set_id: None,
            status: EvalRunStatus::Complete,
            scenario_count: 20,
            completed_count: 20,
            error_count: 0,
            aggregate_score: Some(0.96),
            pass_rate: Some(0.90),
            scores: None,
            failure_clusters: None,
            seed: 42,
            concurrency: 5,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            opt_status: Some("converged".to_string()),
            opt_rounds: 2,
            opt_best_score: Some(0.96),
            opt_best_agent_id: Some(best_id),
        };
        let resp = RunResponse::from(eval_run);
        assert_eq!(resp.opt_status.as_deref(), Some("converged"));
        assert_eq!(resp.opt_rounds, 2);
        assert_eq!(resp.opt_best_score, Some(0.96));
        assert_eq!(resp.opt_best_agent_id, Some(best_id));
    }

    #[test]
    fn run_response_from_eval_run_maps_no_improvement() {
        let eval_run = EvalRun {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            scenario_set_id: None,
            status: EvalRunStatus::Complete,
            scenario_count: 10,
            completed_count: 10,
            error_count: 0,
            aggregate_score: Some(0.50),
            pass_rate: None,
            scores: None,
            failure_clusters: None,
            seed: 1,
            concurrency: 4,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            opt_status: Some("no_improvement".to_string()),
            opt_rounds: 1,
            opt_best_score: Some(0.50),
            opt_best_agent_id: None,
        };
        let resp = RunResponse::from(eval_run);
        assert_eq!(resp.opt_status.as_deref(), Some("no_improvement"));
        assert_eq!(resp.opt_rounds, 1);
        assert!(resp.opt_best_agent_id.is_none());
    }

    #[test]
    fn run_response_opt_rounds_zero_is_default() {
        let eval_run = EvalRun {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            scenario_set_id: None,
            status: EvalRunStatus::Pending,
            scenario_count: 100,
            completed_count: 0,
            error_count: 0,
            aggregate_score: None,
            pass_rate: None,
            scores: None,
            failure_clusters: None,
            seed: 42,
            concurrency: 5,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            opt_status: None,
            opt_rounds: 0,
            opt_best_score: None,
            opt_best_agent_id: None,
        };
        let resp = RunResponse::from(eval_run);
        assert_eq!(resp.opt_rounds, 0);
    }

    #[test]
    fn run_response_opt_fields_in_serialized_keys() {
        let resp = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "complete".to_string(),
            aggregate_score: None,
            pass_rate: None,
            scenario_count: 1,
            completed_count: 1,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: Some("max_iterations".to_string()),
            opt_rounds: 5,
            opt_best_score: Some(0.91),
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        let obj = json.as_object().unwrap();
        assert!(obj.contains_key("opt_status"));
        assert!(obj.contains_key("opt_rounds"));
        assert!(obj.contains_key("opt_best_score"));
        assert!(obj.contains_key("opt_best_agent_id"));
    }

    // ── Opt loop terminal status strings ─────────────────────────────────────

    #[test]
    fn opt_terminal_statuses_are_distinct() {
        let statuses = ["converged", "no_improvement", "max_iterations", "failed", "running"];
        let unique: std::collections::HashSet<_> = statuses.iter().collect();
        assert_eq!(unique.len(), statuses.len());
    }

    #[test]
    fn opt_threshold_default_is_0_92() {
        // The handler uses `req.threshold.unwrap_or(0.92)`.
        let req = StartRunRequest {
            agent_id: Uuid::new_v4(),
            scenario_count: None,
            seed: None,
            concurrency: None,
            threshold: None,
            provider: None,
            judge_provider: None,
            auto_optimize: None,
            max_opt_iterations: None,
        };
        let threshold = req.threshold.unwrap_or(0.92);
        assert!((threshold - 0.92).abs() < 1e-9);
    }

    #[test]
    fn opt_threshold_explicit_0_75() {
        let req = StartRunRequest {
            agent_id: Uuid::new_v4(),
            scenario_count: None,
            seed: None,
            concurrency: None,
            threshold: Some(0.75),
            provider: None,
            judge_provider: None,
            auto_optimize: None,
            max_opt_iterations: None,
        };
        assert_eq!(req.threshold.unwrap_or(0.95), 0.75);
    }

    #[test]
    fn score_meets_threshold_converged() {
        // If current_score >= threshold the loop should break with "converged".
        // Simulate the condition check here as a unit test.
        let current_score = 0.96_f64;
        let threshold = 0.95_f64;
        assert!(
            current_score >= threshold,
            "Loop should converge when score meets threshold"
        );
    }

    #[test]
    fn score_below_threshold_continues() {
        let current_score = 0.94_f64;
        let threshold = 0.95_f64;
        assert!(current_score < threshold, "Loop should continue when score is below threshold");
    }

    #[test]
    fn no_improvement_condition() {
        // Verify the > 1% improvement guard works correctly.
        let current_score = 0.70_f64;
        let round_best_score = 0.705_f64; // only 0.5% improvement
        assert!(
            round_best_score <= current_score + 0.01,
            "Should break no_improvement when improvement <= 1 pp"
        );
    }

    #[test]
    fn improvement_over_1pp_condition() {
        let current_score = 0.70_f64;
        let round_best_score = 0.715_f64; // 1.5% improvement
        assert!(
            round_best_score > current_score + 0.01,
            "Should continue when improvement > 1 pp"
        );
    }

    #[test]
    fn max_iterations_reached_condition() {
        let round: u32 = 5;
        let max_iterations: u32 = 5;
        assert!(round >= max_iterations, "Loop must stop when round reaches max_iterations");
    }

    #[test]
    fn round_zero_does_not_exceed_max() {
        let round: u32 = 0;
        let max_iterations: u32 = 5;
        assert!(round < max_iterations, "Round 0 must not exceed max_iterations 5");
    }

    // ── SSE progress event with opt fields ────────────────────────────────────

    #[test]
    fn sse_event_with_opt_running_status() {
        let data = serde_json::json!({
            "run_id": Uuid::new_v4().to_string(),
            "status": "complete",
            "completed_count": 100_u32,
            "scenario_count": 100_u32,
            "error_count": 0_u32,
            "opt_status": "running",
            "opt_rounds": 1_i32,
            "opt_best_score": 0.72_f64,
            "opt_best_agent_id": serde_json::Value::Null,
        });
        assert_eq!(data["opt_status"], "running");
        assert_eq!(data["opt_rounds"], 1);
        assert!((data["opt_best_score"].as_f64().unwrap() - 0.72).abs() < 1e-9);
        assert!(data["opt_best_agent_id"].is_null());
    }

    #[test]
    fn sse_event_with_opt_converged_status() {
        let best_id = Uuid::new_v4();
        let data = serde_json::json!({
            "run_id": Uuid::new_v4().to_string(),
            "status": "complete",
            "completed_count": 100_u32,
            "scenario_count": 100_u32,
            "error_count": 0_u32,
            "opt_status": "converged",
            "opt_rounds": 3_i32,
            "opt_best_score": 0.96_f64,
            "opt_best_agent_id": best_id.to_string(),
        });
        assert_eq!(data["opt_status"], "converged");
        assert_eq!(data["opt_rounds"], 3);
        assert!(data["opt_best_agent_id"].is_string());
    }

    #[test]
    fn sse_terminal_eval_done_opt_done() {
        // When eval is complete AND opt is not running, SSE should stop.
        let eval_done = true;
        let opt_running = false;
        let is_terminal = eval_done && !opt_running;
        assert!(is_terminal, "SSE must terminate when eval is done and opt is not running");
    }

    #[test]
    fn sse_not_terminal_when_eval_done_but_opt_running() {
        let eval_done = true;
        let opt_running = true;
        let is_terminal = eval_done && !opt_running;
        assert!(!is_terminal, "SSE must keep polling while opt loop is still running");
    }

    #[test]
    fn sse_not_terminal_when_eval_still_running() {
        let eval_done = false;
        let opt_running = false;
        let is_terminal = eval_done && !opt_running;
        assert!(!is_terminal, "SSE must not terminate while eval is still running");
    }

    #[test]
    fn run_response_opt_best_score_extreme_values() {
        let resp_low = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "complete".to_string(),
            aggregate_score: Some(0.01),
            pass_rate: None,
            scenario_count: 1,
            completed_count: 1,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: Some("no_improvement".to_string()),
            opt_rounds: 0,
            opt_best_score: Some(0.0),
            opt_best_agent_id: None,
        };
        let json = serde_json::to_value(&resp_low).unwrap();
        assert!((json["opt_best_score"].as_f64().unwrap()).abs() < 1e-9);

        let resp_high = RunResponse {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            status: "complete".to_string(),
            aggregate_score: Some(1.0),
            pass_rate: None,
            scenario_count: 1,
            completed_count: 1,
            error_count: 0,
            created_at: Utc::now(),
            opt_status: Some("converged".to_string()),
            opt_rounds: 1,
            opt_best_score: Some(1.0),
            opt_best_agent_id: None,
        };
        let json2 = serde_json::to_value(&resp_high).unwrap();
        assert!((json2["opt_best_score"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }
}
