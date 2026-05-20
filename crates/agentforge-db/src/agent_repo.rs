use crate::db_err;
use agentforge_core::{AgentFile, AgentFileFormat, AgentForgeError, AgentVersion, Result};
use chrono::Utc;
use sqlx::PgPool;
use uuid::Uuid;

pub struct AgentRepo {
    pool: PgPool,
}

impl AgentRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Insert a new agent version. Returns error if SHA already exists.
    pub async fn insert(&self, version: &AgentVersion) -> Result<AgentVersion> {
        let format_str = version.format.to_string();
        let file_content_json = serde_json::to_value(&version.file_content)
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;

        let row = sqlx::query!(
            r#"
            INSERT INTO agent_versions
                (id, name, version, sha, file_content, raw_content, format,
                 promoted, is_champion, changelog, parent_sha, created_at, updated_at)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id, name, version, sha, file_content, raw_content, format,
                      promoted, is_champion, changelog, parent_sha, created_at, updated_at
            "#,
            version.id,
            version.name,
            version.version,
            version.sha,
            file_content_json,
            version.raw_content,
            format_str,
            version.promoted,
            version.is_champion,
            version.changelog,
            version.parent_sha,
            Utc::now(),
            Utc::now(),
        )
        .fetch_one(&self.pool)
        .await
        .map_err(db_err)?;

        self.row_to_version(
            row.id,
            row.name,
            row.version,
            row.sha,
            row.file_content,
            row.raw_content,
            row.format,
            row.promoted,
            row.is_champion,
            row.changelog,
            row.parent_sha,
            row.created_at,
            row.updated_at,
        )
    }

    pub async fn find_by_id(&self, id: Uuid) -> Result<AgentVersion> {
        let row = sqlx::query!(
            r#"
            SELECT id, name, version, sha, file_content, raw_content, format,
                   promoted, is_champion, changelog, parent_sha, created_at, updated_at
            FROM agent_versions
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(db_err)?
        .ok_or_else(|| AgentForgeError::NotFound {
            resource: "AgentVersion",
            id: id.to_string(),
        })?;

        self.row_to_version(
            row.id,
            row.name,
            row.version,
            row.sha,
            row.file_content,
            row.raw_content,
            row.format,
            row.promoted,
            row.is_champion,
            row.changelog,
            row.parent_sha,
            row.created_at,
            row.updated_at,
        )
    }

    pub async fn find_by_sha(&self, sha: &str) -> Result<Option<AgentVersion>> {
        let row = sqlx::query!(
            r#"
            SELECT id, name, version, sha, file_content, raw_content, format,
                   promoted, is_champion, changelog, parent_sha, created_at, updated_at
            FROM agent_versions
            WHERE sha = $1
            "#,
            sha
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(db_err)?;

        match row {
            Some(r) => Ok(Some(self.row_to_version(
                r.id,
                r.name,
                r.version,
                r.sha,
                r.file_content,
                r.raw_content,
                r.format,
                r.promoted,
                r.is_champion,
                r.changelog,
                r.parent_sha,
                r.created_at,
                r.updated_at,
            )?)),
            None => Ok(None),
        }
    }

    pub async fn list_by_name(&self, name: &str) -> Result<Vec<AgentVersion>> {
        let rows = sqlx::query!(
            r#"
            SELECT id, name, version, sha, file_content, raw_content, format,
                   promoted, is_champion, changelog, parent_sha, created_at, updated_at
            FROM agent_versions
            WHERE name = $1
            ORDER BY created_at DESC
            "#,
            name
        )
        .fetch_all(&self.pool)
        .await
        .map_err(db_err)?;

        rows.into_iter()
            .map(|r| {
                self.row_to_version(
                    r.id,
                    r.name,
                    r.version,
                    r.sha,
                    r.file_content,
                    r.raw_content,
                    r.format,
                    r.promoted,
                    r.is_champion,
                    r.changelog,
                    r.parent_sha,
                    r.created_at,
                    r.updated_at,
                )
            })
            .collect()
    }

    pub async fn list_all(&self, limit: i64, offset: i64) -> Result<Vec<AgentVersion>> {
        let rows = sqlx::query!(
            r#"
            SELECT id, name, version, sha, file_content, raw_content, format,
                   promoted, is_champion, changelog, parent_sha, created_at, updated_at
            FROM agent_versions
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
            limit,
            offset
        )
        .fetch_all(&self.pool)
        .await
        .map_err(db_err)?;

        rows.into_iter()
            .map(|r| {
                self.row_to_version(
                    r.id,
                    r.name,
                    r.version,
                    r.sha,
                    r.file_content,
                    r.raw_content,
                    r.format,
                    r.promoted,
                    r.is_champion,
                    r.changelog,
                    r.parent_sha,
                    r.created_at,
                    r.updated_at,
                )
            })
            .collect()
    }

    pub async fn set_champion(&self, agent_id: Uuid, agent_name: &str) -> Result<()> {
        let mut tx = self.pool.begin().await.map_err(db_err)?;

        // Clear existing champion for this agent name
        sqlx::query!(
            "UPDATE agent_versions SET is_champion = FALSE WHERE name = $1",
            agent_name
        )
        .execute(&mut *tx)
        .await
        .map_err(db_err)?;

        // Set the new champion
        sqlx::query!(
            "UPDATE agent_versions SET is_champion = TRUE, promoted = TRUE WHERE id = $1",
            agent_id
        )
        .execute(&mut *tx)
        .await
        .map_err(db_err)?;

        tx.commit().await.map_err(db_err)?;
        Ok(())
    }

    pub async fn update_changelog(&self, agent_id: Uuid, changelog: &str) -> Result<()> {
        sqlx::query!(
            "UPDATE agent_versions SET changelog = $1 WHERE id = $2",
            changelog,
            agent_id
        )
        .execute(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(())
    }

    pub async fn delete(&self, id: Uuid) -> Result<bool> {
        let result = sqlx::query("DELETE FROM agent_versions WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(db_err)?;
        Ok(result.rows_affected() > 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn row_to_version(
        &self,
        id: Uuid,
        name: String,
        version: String,
        sha: String,
        file_content: serde_json::Value,
        raw_content: String,
        format: String,
        promoted: bool,
        is_champion: bool,
        changelog: Option<String>,
        parent_sha: Option<String>,
        created_at: chrono::DateTime<Utc>,
        updated_at: chrono::DateTime<Utc>,
    ) -> Result<AgentVersion> {
        use std::str::FromStr;
        let parsed_format = AgentFileFormat::from_str(&format)?;
        let parsed_file: AgentFile = serde_json::from_value(file_content)
            .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;

        Ok(AgentVersion {
            id,
            name,
            version,
            sha,
            file_content: parsed_file,
            raw_content,
            format: parsed_format,
            promoted,
            is_champion,
            changelog,
            parent_sha,
            created_at,
            updated_at,
        })
    }
}
