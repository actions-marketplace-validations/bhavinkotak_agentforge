use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use agentforge_core::{AgentVersion, Scenario};
use agentforge_db::{agent_repo::AgentRepo, scenario_repo::ScenarioRepo};
use agentforge_parser::parse_agent_file;

use crate::{
    error::{ApiError, ApiResult},
    state::AppState,
};

#[derive(Debug, Deserialize)]
pub struct CreateAgentRequest {
    /// Raw agent file content (YAML or JSON)
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub sha: String,
    pub parent_sha: Option<String>,
    pub format: String,
    pub promoted: bool,
    pub is_champion: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl From<AgentVersion> for AgentResponse {
    fn from(v: AgentVersion) -> Self {
        Self {
            id: v.id,
            name: v.name,
            version: v.version,
            sha: v.sha,
            parent_sha: v.parent_sha,
            format: v.format.to_string(),
            promoted: v.promoted,
            is_champion: v.is_champion,
            created_at: v.created_at,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ListAgentsQuery {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// POST /agents
pub async fn create_agent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateAgentRequest>,
) -> ApiResult<(StatusCode, Json<AgentResponse>)> {
    // Parse and validate
    let parsed = parse_agent_file(&req.content)
        .map_err(|e| ApiError::bad_request(format!("Parse error: {e}")))?;

    let validation = agentforge_parser::validate_agent_file(&parsed.agent);
    let critical_errors: Vec<_> = validation
        .errors
        .iter()
        .filter(|e| e.severity == agentforge_core::LintSeverity::Error)
        .collect();
    if !critical_errors.is_empty() {
        let msgs: Vec<_> = critical_errors.iter().map(|e| e.message.clone()).collect();
        return Err(ApiError::bad_request(format!(
            "Validation failed: {}",
            msgs.join("; ")
        )));
    }

    let agent_version = agentforge_parser::to_agent_version(parsed);
    let repo = AgentRepo::new(state.db.clone());

    // Check for duplicate SHA (idempotent upsert)
    if let Some(existing) = repo
        .find_by_sha(&agent_version.sha)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
    {
        return Ok((StatusCode::OK, Json(existing.into())));
    }

    let saved = repo
        .insert(&agent_version)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(saved.into())))
}

/// GET /agents/:id
pub async fn get_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<AgentResponse>> {
    let repo = AgentRepo::new(state.db.clone());
    let agent = repo.find_by_id(id).await.map_err(|e| match e {
        agentforge_core::AgentForgeError::NotFound { .. } => {
            ApiError::not_found(format!("Agent {id} not found"))
        }
        other => ApiError::internal(other.to_string()),
    })?;
    Ok(Json(agent.into()))
}

/// GET /agents
pub async fn list_agents(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListAgentsQuery>,
) -> ApiResult<Json<Vec<AgentResponse>>> {
    let repo = AgentRepo::new(state.db.clone());
    let limit = params.limit.unwrap_or(50).min(200);
    let offset = params.offset.unwrap_or(0);
    let agents = repo
        .list_all(limit, offset)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(agents.into_iter().map(Into::into).collect()))
}

/// DELETE /agents/:id
///
/// Deletes the agent version. Returns 404 if not found, 409 if the agent is
/// currently the champion (promote a different version first).
pub async fn delete_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> ApiResult<StatusCode> {
    let repo = AgentRepo::new(state.db.clone());
    // Guard: refuse to delete the current champion
    let agent = repo.find_by_id(id).await.map_err(|e| match e {
        agentforge_core::AgentForgeError::NotFound { .. } => {
            ApiError::not_found(format!("Agent {id} not found"))
        }
        other => ApiError::internal(other.to_string()),
    })?;
    if agent.is_champion {
        return Err(ApiError::conflict(
            "Cannot delete the current champion. Promote a different version first.",
        ));
    }
    let deleted = repo
        .delete(id)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::not_found(format!("Agent {id} not found")))
    }
}

// ─── PATCH /agents/:id ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct PatchAgentRequest {
    /// When `true`, designates this version as the current champion.
    /// All other versions of the same agent name are demoted automatically.
    pub is_champion: Option<bool>,
    /// Update the human-readable changelog entry for this version.
    pub changelog: Option<String>,
}

/// PATCH /agents/:id — update mutable metadata for an agent version.
///
/// Supported mutations:
/// - `is_champion: true` — crown this version as the champion without running the full gatekeeper
///   pipeline (useful for bootstrapping or manual overrides).
/// - `changelog` — update the changelog text recorded against this version.
pub async fn patch_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<PatchAgentRequest>,
) -> ApiResult<Json<AgentResponse>> {
    let repo = AgentRepo::new(state.db.clone());

    let agent = repo.find_by_id(id).await.map_err(|e| match e {
        agentforge_core::AgentForgeError::NotFound { .. } => {
            ApiError::not_found(format!("Agent {id} not found"))
        }
        other => ApiError::internal(other.to_string()),
    })?;

    if req.is_champion == Some(true) {
        repo.set_champion(id, &agent.name)
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?;
    }

    if let Some(ref changelog) = req.changelog {
        repo.update_changelog(id, changelog)
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?;
    }

    // Return the updated record
    let updated = repo
        .find_by_id(id)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(updated.into()))
}

// ─── GET /agents/:id/scenarios ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ListScenariosQuery {
    pub limit: Option<i64>,
}

/// GET /agents/:id/scenarios — list generated test scenarios for an agent version.
///
/// Returns up to `limit` scenarios (default 50, max 500).
pub async fn list_agent_scenarios(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(params): Query<ListScenariosQuery>,
) -> ApiResult<Json<Vec<Scenario>>> {
    // Verify the agent exists first so we return 404 rather than an empty list
    let agent_repo = AgentRepo::new(state.db.clone());
    agent_repo.find_by_id(id).await.map_err(|e| match e {
        agentforge_core::AgentForgeError::NotFound { .. } => {
            ApiError::not_found(format!("Agent {id} not found"))
        }
        other => ApiError::internal(other.to_string()),
    })?;

    let scenario_repo = ScenarioRepo::new(state.db.clone());
    let limit = params.limit.unwrap_or(50).min(500);
    let scenarios = scenario_repo
        .list_by_agent(id, limit)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(scenarios))
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PatchAgentRequest deserialization ─────────────────────────────────────

    /// Both optional fields present.
    #[test]
    fn patch_request_deserializes_both_fields() {
        let json = serde_json::json!({
            "is_champion": true,
            "changelog": "Bumped temperature to 0.3 for better diversity."
        });
        let req: PatchAgentRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.is_champion, Some(true));
        assert_eq!(
            req.changelog.as_deref(),
            Some("Bumped temperature to 0.3 for better diversity.")
        );
    }

    /// Only `is_champion` provided.
    #[test]
    fn patch_request_champion_only() {
        let json = serde_json::json!({ "is_champion": true });
        let req: PatchAgentRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.is_champion, Some(true));
        assert!(req.changelog.is_none());
    }

    /// Only `changelog` provided.
    #[test]
    fn patch_request_changelog_only() {
        let json = serde_json::json!({ "changelog": "Initial champion." });
        let req: PatchAgentRequest = serde_json::from_value(json).unwrap();
        assert!(req.is_champion.is_none());
        assert_eq!(req.changelog.as_deref(), Some("Initial champion."));
    }

    /// Empty JSON object — all fields absent.
    #[test]
    fn patch_request_empty_body_is_valid() {
        let json = serde_json::json!({});
        let req: PatchAgentRequest = serde_json::from_value(json).unwrap();
        assert!(req.is_champion.is_none());
        assert!(req.changelog.is_none());
    }

    /// `is_champion: false` — explicitly demoting should also parse correctly.
    #[test]
    fn patch_request_is_champion_false() {
        let json = serde_json::json!({ "is_champion": false });
        let req: PatchAgentRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.is_champion, Some(false));
    }

    // ── ListScenariosQuery limit clamping logic ───────────────────────────────

    /// Default (absent limit) should fall back to 50.
    #[test]
    fn list_scenarios_query_default_limit_is_50() {
        let query: ListScenariosQuery = serde_json::from_value(serde_json::json!({})).unwrap();
        let limit = query.limit.unwrap_or(50).min(500);
        assert_eq!(limit, 50);
    }

    /// A limit above 500 must be clamped to 500.
    #[test]
    fn list_scenarios_query_clamps_limit_at_500() {
        let query: ListScenariosQuery =
            serde_json::from_value(serde_json::json!({ "limit": 9999 })).unwrap();
        let limit = query.limit.unwrap_or(50).min(500);
        assert_eq!(limit, 500);
    }

    /// Limit of exactly 500 passes through unchanged.
    #[test]
    fn list_scenarios_query_limit_500_not_clamped() {
        let query: ListScenariosQuery =
            serde_json::from_value(serde_json::json!({ "limit": 500 })).unwrap();
        let limit = query.limit.unwrap_or(50).min(500);
        assert_eq!(limit, 500);
    }

    /// A small limit value passes through unchanged.
    #[test]
    fn list_scenarios_query_small_limit_unchanged() {
        let query: ListScenariosQuery =
            serde_json::from_value(serde_json::json!({ "limit": 10 })).unwrap();
        let limit = query.limit.unwrap_or(50).min(500);
        assert_eq!(limit, 10);
    }

    // ── AgentResponse From<AgentVersion> conversion ───────────────────────────

    #[test]
    fn agent_response_from_agent_version_maps_fields() {
        use agentforge_core::{AgentFile, AgentFileFormat, ModelConfig, ModelProvider};

        let id = Uuid::new_v4();
        let now = chrono::Utc::now();
        let dummy_agent_file = AgentFile {
            agentforge_schema_version: "1".to_string(),
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            model: ModelConfig {
                provider: ModelProvider::Openai,
                model_id: "gpt-4o".to_string(),
                temperature: Some(0.2),
                max_tokens: None,
                top_p: None,
            },
            system_prompt: "You are a test agent.".to_string(),
            tools: vec![],
            output_schema: None,
            constraints: vec![],
            eval_hints: None,
            metadata: None,
        };

        let av = AgentVersion {
            id,
            name: "test-agent".to_string(),
            version: "2.0.0".to_string(),
            sha: "abc123def456".to_string(),
            raw_content: String::new(),
            format: AgentFileFormat::NativeYaml,
            file_content: dummy_agent_file,
            promoted: false,
            is_champion: true,
            changelog: None,
            parent_sha: None,
            created_at: now,
            updated_at: now,
        };

        let resp = AgentResponse::from(av);
        assert_eq!(resp.id, id);
        assert_eq!(resp.name, "test-agent");
        assert_eq!(resp.version, "2.0.0");
        assert_eq!(resp.sha, "abc123def456");
        assert!(resp.is_champion);
        assert!(!resp.promoted);
        assert_eq!(resp.format, "native_yaml");
    }

    // ── CreateAgentRequest deserialization ────────────────────────────────────

    #[test]
    fn create_agent_request_deserializes_content_field() {
        let json = serde_json::json!({
            "content": "agentforge_schema_version: \"1\"\nname: my-agent\nversion: \"1.0.0\""
        });
        let req: CreateAgentRequest = serde_json::from_value(json).unwrap();
        assert!(req.content.contains("my-agent"));
    }

    // ── ListAgentsQuery pagination defaults ───────────────────────────────────

    #[test]
    fn list_agents_query_default_limit_is_50() {
        let q: ListAgentsQuery = serde_json::from_value(serde_json::json!({})).unwrap();
        let limit = q.limit.unwrap_or(50).min(200);
        assert_eq!(limit, 50);
    }

    #[test]
    fn list_agents_query_clamps_limit_at_200() {
        let q: ListAgentsQuery =
            serde_json::from_value(serde_json::json!({ "limit": 9999 })).unwrap();
        let limit = q.limit.unwrap_or(50).min(200);
        assert_eq!(limit, 200);
    }
}
