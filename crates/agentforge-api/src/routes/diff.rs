use axum::{
    extract::{Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use agentforge_core::AgentVersion;
use agentforge_db::agent_repo::AgentRepo;

use crate::{
    error::{ApiError, ApiResult},
    state::AppState,
};

#[derive(Debug, Deserialize)]
pub struct DiffQuery {
    pub v1: Uuid,
    pub v2: Uuid,
}

#[derive(Debug, Serialize)]
pub struct DiffResponse {
    pub v1: AgentSummary,
    pub v2: AgentSummary,
    pub system_prompt_diff: Option<String>,
    pub tool_changes: ToolChanges,
    pub constraint_changes: ConstraintChanges,
}

#[derive(Debug, Serialize)]
pub struct AgentSummary {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub sha: String,
    pub is_champion: bool,
}

#[derive(Debug, Serialize)]
pub struct ToolChanges {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub modified: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ConstraintChanges {
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

impl From<AgentVersion> for AgentSummary {
    fn from(v: AgentVersion) -> Self {
        Self {
            id: v.id,
            name: v.name,
            version: v.version,
            sha: v.sha,
            is_champion: v.is_champion,
        }
    }
}

/// GET /diff?v1=<uuid>&v2=<uuid>
pub async fn get_diff(
    State(state): State<Arc<AppState>>,
    Query(params): Query<DiffQuery>,
) -> ApiResult<Json<DiffResponse>> {
    let repo = AgentRepo::new(state.db.clone());

    let v1 = repo.find_by_id(params.v1).await.map_err(|e| match e {
        agentforge_core::AgentForgeError::NotFound { .. } => {
            ApiError::not_found(format!("Agent version {} not found", params.v1))
        }
        other => ApiError::internal(other.to_string()),
    })?;

    let v2 = repo.find_by_id(params.v2).await.map_err(|e| match e {
        agentforge_core::AgentForgeError::NotFound { .. } => {
            ApiError::not_found(format!("Agent version {} not found", params.v2))
        }
        other => ApiError::internal(other.to_string()),
    })?;

    let diff = compute_diff(&v1, &v2);
    Ok(Json(diff))
}

/// Compute a unified-style line diff between two strings.
/// Returns lines prefixed with `+` (added), `-` (removed), or ` ` (context).
/// Hunks are separated by `@@ ... @@` headers. Returns `None` if identical.
fn unified_diff(old: &str, new: &str) -> Option<String> {
    if old == new {
        return None;
    }

    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    let m = old_lines.len();
    let n = new_lines.len();

    // LCS DP table (m+1) × (n+1)
    let mut dp = vec![vec![0u32; n + 1]; m + 1];
    for i in (0..m).rev() {
        for j in (0..n).rev() {
            dp[i][j] = if old_lines[i] == new_lines[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    // Build edit list: (' ', context), ('-', removed), ('+', added)
    let mut edits: Vec<(char, &str)> = Vec::with_capacity(m + n);
    let (mut i, mut j) = (0, 0);
    while i < m || j < n {
        if i < m && j < n && old_lines[i] == new_lines[j] {
            edits.push((' ', old_lines[i]));
            i += 1;
            j += 1;
        } else if i < m && (j >= n || dp[i + 1][j] >= dp[i][j + 1]) {
            edits.push(('-', old_lines[i]));
            i += 1;
        } else {
            edits.push(('+', new_lines[j]));
            j += 1;
        }
    }

    // Collect positions of actual changes
    let changed: Vec<usize> = edits
        .iter()
        .enumerate()
        .filter(|(_, (c, _))| *c != ' ')
        .map(|(idx, _)| idx)
        .collect();

    if changed.is_empty() {
        return None;
    }

    // Group changes into hunks with 3 lines of context
    const CTX: usize = 3;
    let mut hunks: Vec<(usize, usize)> = Vec::new();
    let mut hunk_start = changed[0].saturating_sub(CTX);
    let mut hunk_end = (changed[0] + CTX + 1).min(edits.len());
    for &pos in &changed[1..] {
        if pos.saturating_sub(CTX) <= hunk_end {
            hunk_end = (pos + CTX + 1).min(edits.len());
        } else {
            hunks.push((hunk_start, hunk_end));
            hunk_start = pos.saturating_sub(CTX);
            hunk_end = (pos + CTX + 1).min(edits.len());
        }
    }
    hunks.push((hunk_start, hunk_end));

    let mut result = String::new();
    for (start, end) in hunks {
        result.push_str("@@ change @@\n");
        for (action, line) in &edits[start..end] {
            result.push(*action);
            result.push(' ');
            result.push_str(line);
            result.push('\n');
        }
    }
    Some(result)
}

fn compute_diff(v1: &AgentVersion, v2: &AgentVersion) -> DiffResponse {
    let prompt1 = v1.file_content.system_prompt.as_str();
    let prompt2 = v2.file_content.system_prompt.as_str();
    let system_prompt_diff = unified_diff(prompt1, prompt2);

    // Tool changes
    let tools1: Vec<String> = v1
        .file_content
        .tools
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let tools2: Vec<String> = v2
        .file_content
        .tools
        .iter()
        .map(|t| t.name.clone())
        .collect();

    let added_tools: Vec<String> = tools2
        .iter()
        .filter(|t| !tools1.contains(t))
        .cloned()
        .collect();
    let removed_tools: Vec<String> = tools1
        .iter()
        .filter(|t| !tools2.contains(t))
        .cloned()
        .collect();

    // Constraint changes
    let constraints1: Vec<String> = v1.file_content.constraints.clone();
    let constraints2: Vec<String> = v2.file_content.constraints.clone();

    let added_constraints: Vec<String> = constraints2
        .iter()
        .filter(|c| !constraints1.contains(c))
        .cloned()
        .collect();
    let removed_constraints: Vec<String> = constraints1
        .iter()
        .filter(|c| !constraints2.contains(c))
        .cloned()
        .collect();

    DiffResponse {
        v1: v1.clone().into(),
        v2: v2.clone().into(),
        system_prompt_diff,
        tool_changes: ToolChanges {
            added: added_tools,
            removed: removed_tools,
            modified: vec![], // Would need deep comparison
        },
        constraint_changes: ConstraintChanges {
            added: added_constraints,
            removed: removed_constraints,
        },
    }
}
