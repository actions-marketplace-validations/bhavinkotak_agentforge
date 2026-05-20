use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use agentforge_core::AgentVersion;
use agentforge_db::agent_repo::AgentRepo;
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
