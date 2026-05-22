use std::sync::Arc;

use agentforge_api::{router, AppState};
use agentforge_db::create_pool;
use agentforge_gatekeeper::GatekeeperConfig;
use agentforge_observability::build_exporter;
use agentforge_optimizer::OptimizerConfig;
use agentforge_runner::{AnthropicClient, LlmClient, NvidiaClient, OpenAiClient};
use agentforge_scorer::ScorerConfig;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();

    let log_level = std::env::var("AGENTFORGE_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    tracing_subscriber::registry()
        .with(EnvFilter::new(log_level))
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable is required");
    let db = create_pool(&database_url).await?;
    agentforge_db::run_migrations(&db).await?;

    let llm_client: Arc<dyn LlmClient> = {
        let provider =
            std::env::var("AGENTFORGE_JUDGE_PROVIDER").unwrap_or_else(|_| "openai".to_string());
        match provider.as_str() {
            "anthropic" => Arc::new(
                AnthropicClient::from_env()
                    .expect("ANTHROPIC_API_KEY must be set when using anthropic provider"),
            ) as Arc<dyn LlmClient>,
            "nvidia" => Arc::new(
                NvidiaClient::from_env()
                    .expect("NVIDIA_API_KEY must be set when using nvidia provider"),
            ) as Arc<dyn LlmClient>,
            _ => Arc::new(
                OpenAiClient::from_env()
                    .expect("OPENAI_API_KEY must be set when using openai provider"),
            ) as Arc<dyn LlmClient>,
        }
    };

    // Derive scorer judge credentials from the same provider selection so that
    // switching AGENTFORGE_JUDGE_PROVIDER also routes the scorer correctly.
    let judge_provider =
        std::env::var("AGENTFORGE_JUDGE_PROVIDER").unwrap_or_else(|_| "openai".to_string());
    let (default_judge_base_url, default_judge_api_key) = match judge_provider.as_str() {
        "anthropic" => (
            "https://api.anthropic.com/v1".to_string(),
            std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
        ),
        "nvidia" => (
            "https://integrate.api.nvidia.com/v1".to_string(),
            std::env::var("NVIDIA_API_KEY").unwrap_or_default(),
        ),
        _ => (
            "https://api.openai.com/v1".to_string(),
            std::env::var("OPENAI_API_KEY").unwrap_or_default(),
        ),
    };
    let scorer_config = ScorerConfig {
        judge_model: std::env::var("AGENTFORGE_JUDGE_MODEL")
            .unwrap_or_else(|_| "gpt-4o".to_string()),
        judge_base_url: std::env::var("AGENTFORGE_JUDGE_BASE_URL")
            .unwrap_or(default_judge_base_url),
        judge_api_key: std::env::var("AGENTFORGE_JUDGE_API_KEY").unwrap_or(default_judge_api_key),
        ..Default::default()
    };

    let gatekeeper_config = GatekeeperConfig::default();
    let trace_exporter: Arc<dyn agentforge_observability::TraceExporter> =
        Arc::from(build_exporter());

    // Optimizer uses the same judge LLM (prompt rewrites need a capable model)
    let optimizer_config = OptimizerConfig {
        llm_base_url: scorer_config.judge_base_url.clone(),
        llm_api_key: scorer_config.judge_api_key.clone(),
        llm_model: scorer_config.judge_model.clone(),
        min_variants: 3,
        max_variants: 5,
        few_shot_min_traces: 3, // lower bar so we actually use passing traces
    };

    let max_concurrent_runs: i64 = std::env::var("AGENTFORGE_MAX_CONCURRENT_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let max_scenarios: u32 = std::env::var("AGENTFORGE_MAX_SCENARIOS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    let api_key: Option<String> = std::env::var("AGENTFORGE_API_KEY").ok();
    if api_key.is_some() {
        tracing::info!("API key authentication enabled");
    } else {
        tracing::warn!("AGENTFORGE_API_KEY is not set — running in unauthenticated mode");
    }

    let state = Arc::new(AppState {
        db,
        llm_client,
        scorer_config,
        optimizer_config,
        gatekeeper_config,
        trace_exporter,
        active_runs: Arc::new(std::sync::atomic::AtomicI64::new(0)),
        max_concurrent_runs,
        max_scenarios,
        api_key,
    });

    let host = std::env::var("AGENTFORGE_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("AGENTFORGE_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr = format!("{host}:{port}");

    tracing::info!("AgentForge API listening on {addr}");

    let app = router(state)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::cors::CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
