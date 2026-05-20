use agentforge_db::PgPool;
use agentforge_gatekeeper::GatekeeperConfig;
use agentforge_observability::TraceExporter;
use agentforge_runner::LlmClient;
use agentforge_scorer::ScorerConfig;
use std::sync::{atomic::AtomicI64, Arc};

/// Shared application state injected into all route handlers.
pub struct AppState {
    pub db: PgPool,
    pub llm_client: Arc<dyn LlmClient>,
    pub scorer_config: ScorerConfig,
    pub gatekeeper_config: GatekeeperConfig,
    pub trace_exporter: Arc<dyn TraceExporter>,
    /// Counts currently active evaluation runs (background tasks).
    /// `POST /runs` is rejected with 429 when this exceeds `AGENTFORGE_MAX_CONCURRENT_RUNS`
    /// (default: 10). Prevents accidental runaway LLM cost from flooding the queue.
    pub active_runs: Arc<AtomicI64>,
    /// Maximum number of concurrently active evaluation runs.
    pub max_concurrent_runs: i64,
    /// Maximum number of scenarios allowed per eval run (`AGENTFORGE_MAX_SCENARIOS`, default 2000).
    pub max_scenarios: u32,
    /// Optional Bearer token for API authentication (`AGENTFORGE_API_KEY`).
    /// When set, all `/api/v1/*` endpoints require `Authorization: Bearer <key>`.
    /// When `None`, the server operates in unauthenticated development mode.
    pub api_key: Option<String>,
}
