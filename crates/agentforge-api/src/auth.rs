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
