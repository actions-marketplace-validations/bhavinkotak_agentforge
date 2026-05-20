use axum::{
    http::StatusCode,
    middleware,
    routing::{delete, get, patch, post},
    Router,
};
use std::sync::Arc;

mod auth;
mod error;
mod routes;
mod state;

pub use state::AppState;

/// Liveness probe — always returns 200. Exempt from API key authentication.
async fn health() -> StatusCode {
    StatusCode::OK
}

pub fn router(state: Arc<AppState>) -> Router {
    let api_routes = Router::new()
        // Agent endpoints
        .route("/agents", post(routes::agents::create_agent))
        .route("/agents", get(routes::agents::list_agents))
        .route("/agents/:id", get(routes::agents::get_agent))
        .route("/agents/:id", delete(routes::agents::delete_agent))
        .route("/agents/:id", patch(routes::agents::patch_agent))
        .route(
            "/agents/:id/scenarios",
            get(routes::agents::list_agent_scenarios),
        )
        // Eval run endpoints
        .route("/runs", get(routes::runs::list_runs))
        .route("/runs", post(routes::runs::start_run))
        .route("/runs/:id", get(routes::runs::get_run))
        .route("/runs/:id", delete(routes::runs::cancel_run))
        .route("/runs/:id/scorecard", get(routes::runs::get_scorecard))
        .route("/runs/:id/traces", get(routes::runs::list_traces))
        .route("/runs/:id/progress", get(routes::runs::run_progress))
        // Diff and promote
        .route("/diff", get(routes::diff::get_diff))
        .route("/promote/:run_id", post(routes::promote::promote_run))
        // Shadow / online eval
        .route("/shadow-runs", post(routes::shadow::start_shadow_run))
        .route("/shadow-runs/:id", get(routes::shadow::get_shadow_run))
        // Fine-tune export
        .route("/exports/finetune", post(routes::finetune::start_export))
        .route("/exports/finetune/:id", get(routes::finetune::get_export))
        // Benchmark comparison
        .route("/benchmarks", post(routes::benchmarks::start_benchmark))
        .route("/benchmarks/:id", get(routes::benchmarks::get_benchmark))
        // Apply optional Bearer-token auth to all /api/v1/* routes
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ));

    Router::new()
        .route("/health", get(health))
        .nest("/api/v1", api_routes)
        .with_state(state)
}
