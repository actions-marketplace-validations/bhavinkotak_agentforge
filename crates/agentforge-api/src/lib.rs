use axum::{
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;

mod error;
mod routes;
mod state;

pub use state::AppState;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        // Agent endpoints
        .route("/agents", post(routes::agents::create_agent))
        .route("/agents", get(routes::agents::list_agents))
        .route("/agents/:id", get(routes::agents::get_agent))
        .route("/agents/:id", delete(routes::agents::delete_agent))
        // Eval run endpoints
        .route("/runs", get(routes::runs::list_runs))
        .route("/runs", post(routes::runs::start_run))
        .route("/runs/:id", get(routes::runs::get_run))
        .route("/runs/:id", delete(routes::runs::cancel_run))
        .route("/runs/:id/scorecard", get(routes::runs::get_scorecard))
        .route("/runs/:id/traces", get(routes::runs::list_traces))
        // Diff and promote
        .route("/diff", get(routes::diff::get_diff))
        .route("/promote/:run_id", post(routes::promote::promote_run))
        // v2: Shadow / online eval
        .route("/shadow-runs", post(routes::shadow::start_shadow_run))
        .route("/shadow-runs/:id", get(routes::shadow::get_shadow_run))
        // v2: Fine-tune export
        .route("/exports/finetune", post(routes::finetune::start_export))
        .route("/exports/finetune/:id", get(routes::finetune::get_export))
        // v2: Benchmark comparison
        .route("/benchmarks", post(routes::benchmarks::start_benchmark))
        .route("/benchmarks/:id", get(routes::benchmarks::get_benchmark))
        .with_state(state)
}
