use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::state::AppState;

/// Optional Bearer-token authentication middleware.
///
/// If `AGENTFORGE_API_KEY` is set in the environment (loaded into `AppState.api_key`),
/// every request must include an `Authorization: Bearer <key>` header matching that value.
/// Requests with a missing or wrong token are rejected with HTTP 401.
///
/// When `api_key` is `None` (env var not set), all requests are allowed through — this
/// is the expected behaviour in local development.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if let Some(ref expected_key) = state.api_key {
        let auth_header = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        let is_valid = auth_header
            .and_then(|h| h.strip_prefix("Bearer "))
            .map(|token| token == expected_key)
            .unwrap_or(false);

        if !is_valid {
            return Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header("WWW-Authenticate", r#"Bearer realm="agentforge""#)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{"error":{"code":"UNAUTHORIZED","message":"Valid Bearer token required"}}"#,
                ))
                .unwrap();
        }
    }

    next.run(request).await
}
// ─── helpers ────────────────────────────────────────────────────────────────

/// Extract a valid Bearer token from an `Authorization` header value.
/// Returns `Some(token)` if the value starts with exactly `"Bearer "` (with one
/// space), `None` otherwise.  Exposed for unit testing only.
#[cfg(test)]
pub(crate) fn extract_bearer_token(header_value: &str) -> Option<&str> {
    header_value.strip_prefix("Bearer ")
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request, middleware, routing::get, Router};
    use std::sync::{
        atomic::{AtomicI64, Ordering},
        Arc,
    };
    use tower::ServiceExt; // for `oneshot`

    // ── Minimal stub types required to construct AppState without a real DB ───

    struct StubLlmClient;

    #[async_trait::async_trait]
    impl agentforge_runner::LlmClient for StubLlmClient {
        async fn complete(
            &self,
            _req: agentforge_runner::LlmRequest,
        ) -> agentforge_core::Result<agentforge_runner::LlmResponse> {
            Err(agentforge_core::AgentForgeError::ConfigError(
                "test stub — not called".to_string(),
            ))
        }
        fn provider_name(&self) -> &str {
            "stub"
        }
        fn model_id(&self) -> &str {
            "stub-model"
        }
    }

    /// Build an `AppState` whose DB pool is lazy (no actual connection is made)
    /// so auth-only tests can run without a Postgres server.
    fn make_test_state(api_key: Option<String>) -> Arc<AppState> {
        // In CI, DATABASE_URL points to the real test postgres. Locally it falls
        // back to a stub URL — the lazy pool only connects if a query is executed,
        // and none of the auth-middleware tests issue any DB queries.
        let db_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://stub:stub@localhost/stub_unused".to_string());
        let db = sqlx::PgPool::connect_lazy(&db_url).unwrap();
        Arc::new(AppState {
            db,
            llm_client: Arc::new(StubLlmClient),
            scorer_config: agentforge_scorer::ScorerConfig::default(),
            optimizer_config: agentforge_optimizer::OptimizerConfig::default(),
            gatekeeper_config: agentforge_gatekeeper::GatekeeperConfig::default(),
            trace_exporter: Arc::new(agentforge_observability::NoopExporter),
            active_runs: Arc::new(AtomicI64::new(0)),
            max_concurrent_runs: 10,
            max_scenarios: 2000,
            api_key,
        })
    }

    /// A simple test router that wraps a GET /test handler behind auth_middleware.
    fn test_router(api_key: Option<String>) -> Router {
        let state = make_test_state(api_key);
        Router::new()
            .route("/test", get(|| async { axum::http::StatusCode::OK }))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .with_state(state)
    }

    // ── Test 1: no api_key configured → all requests pass through ────────────

    #[tokio::test]
    async fn auth_allows_all_when_no_api_key() {
        let app = test_router(None);
        let req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── Test 2: api_key set, no Authorization header → 401 ───────────────────

    #[tokio::test]
    async fn auth_rejects_missing_auth_header() {
        let app = test_router(Some("secret-key-123".to_string()));
        let req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── Test 3: wrong Bearer token → 401 ─────────────────────────────────────

    #[tokio::test]
    async fn auth_rejects_wrong_bearer_token() {
        let app = test_router(Some("correct-key".to_string()));
        let req = Request::builder()
            .uri("/test")
            .header("Authorization", "Bearer wrong-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── Test 4: correct Bearer token → 200 ───────────────────────────────────

    #[tokio::test]
    async fn auth_allows_valid_bearer_token() {
        let key = "my-super-secret-api-key";
        let app = test_router(Some(key.to_string()));
        let req = Request::builder()
            .uri("/test")
            .header("Authorization", format!("Bearer {key}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── Test 5: non-Bearer scheme (Basic) → 401 ───────────────────────────────

    #[tokio::test]
    async fn auth_rejects_non_bearer_scheme() {
        let app = test_router(Some("correct-key".to_string()));
        let req = Request::builder()
            .uri("/test")
            .header("Authorization", "Basic dXNlcjpwYXNz")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── Test 6: 401 includes WWW-Authenticate header ──────────────────────────

    #[tokio::test]
    async fn auth_401_includes_www_authenticate_header() {
        let app = test_router(Some("key".to_string()));
        let req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let www_auth = resp
            .headers()
            .get("WWW-Authenticate")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            www_auth.contains("Bearer"),
            "WWW-Authenticate should contain 'Bearer', got: {www_auth}"
        );
        assert!(
            www_auth.contains("agentforge"),
            "WWW-Authenticate should contain realm 'agentforge', got: {www_auth}"
        );
    }

    // ── Test 7: 401 body is valid JSON with expected error shape ──────────────

    #[tokio::test]
    async fn auth_401_response_body_has_correct_json() {
        let app = test_router(Some("key".to_string()));
        let req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "UNAUTHORIZED");
        assert!(!body["error"]["message"].as_str().unwrap_or("").is_empty());
    }

    // ── Test 8: token with extra leading space is rejected ────────────────────

    #[tokio::test]
    async fn auth_rejects_token_with_leading_space() {
        let app = test_router(Some("correct-key".to_string()));
        // "Bearer  correct-key" (double space) is NOT a valid match
        let req = Request::builder()
            .uri("/test")
            .header("Authorization", "Bearer  correct-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── Test 9: empty string api_key still enforces auth ─────────────────────

    #[tokio::test]
    async fn auth_enforces_even_for_empty_string_key() {
        // An empty string api_key is treated as "key is set" (Some(""))
        let app = test_router(Some(String::new()));
        // Bearer with the empty token value — the actual expected token IS ""
        let req = Request::builder()
            .uri("/test")
            .header("Authorization", "Bearer ")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // "Bearer " strips the prefix and leaves "" which equals the empty key
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── Test 10: pure helper — extract_bearer_token ───────────────────────────

    #[test]
    fn extract_bearer_valid() {
        assert_eq!(extract_bearer_token("Bearer abc123"), Some("abc123"));
    }

    #[test]
    fn extract_bearer_missing_prefix() {
        assert_eq!(extract_bearer_token("abc123"), None);
    }

    #[test]
    fn extract_bearer_wrong_scheme() {
        assert_eq!(extract_bearer_token("Basic abc123"), None);
    }

    #[test]
    fn extract_bearer_double_space() {
        // "Bearer  key" strips prefix "Bearer " and yields " key" — a different token
        assert_eq!(extract_bearer_token("Bearer  key"), Some(" key"));
    }

    // ── Test 11: active_runs counter is unaffected by auth checks ─────────────

    #[tokio::test]
    async fn auth_does_not_modify_active_runs_counter() {
        let state = make_test_state(Some("key".to_string()));
        let initial = state.active_runs.load(Ordering::SeqCst);

        let app = Router::new()
            .route("/test", get(|| async { axum::http::StatusCode::OK }))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .with_state(state.clone());

        let req = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let _ = app.oneshot(req).await.unwrap();

        assert_eq!(
            state.active_runs.load(Ordering::SeqCst),
            initial,
            "auth middleware must not change active_runs"
        );
    }
}
