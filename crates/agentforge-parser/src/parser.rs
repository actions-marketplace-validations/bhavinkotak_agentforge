use crate::{detect::detect_format, formats};
use agentforge_core::{AgentFile, AgentFileFormat, AgentForgeError, AgentVersion, Result};
use chrono::Utc;
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// The result of parsing a raw agent file.
#[derive(Debug, Clone)]
pub struct ParsedAgentFile {
    pub agent: AgentFile,
    pub format: AgentFileFormat,
    pub sha: String,
    pub raw_content: String,
}

/// Parse a raw agent file string into an `AgentFile`.
/// Returns `Err` only for fatal parse errors.
/// Lint warnings are returned via `ValidationResult` from `validate_agent_file`.
pub fn parse_agent_file(content: &str) -> Result<ParsedAgentFile> {
    let format = detect_format(content)?;

    // Parse into serde_json::Value first to allow format-specific normalization
    let value = parse_to_value(content, &format)?;

    let agent = formats::normalize(&format, &value, content)?;
    let sha = compute_sha256(content);

    Ok(ParsedAgentFile {
        agent,
        format,
        sha,
        raw_content: content.to_string(),
    })
}

/// Like `parse_agent_file` but uses an explicitly supplied format instead of auto-detecting.
/// Useful when the file extension is misleading or the user wants to override detection.
pub fn parse_agent_file_with_format(
    content: &str,
    format: AgentFileFormat,
) -> Result<ParsedAgentFile> {
    let value = parse_to_value(content, &format)?;
    let agent = formats::normalize(&format, &value, content)?;
    let sha = compute_sha256(content);
    Ok(ParsedAgentFile {
        agent,
        format,
        sha,
        raw_content: content.to_string(),
    })
}

/// Build an `AgentVersion` from a `ParsedAgentFile`.
pub fn to_agent_version(parsed: ParsedAgentFile) -> AgentVersion {
    let now = Utc::now();
    AgentVersion {
        id: Uuid::new_v4(),
        name: parsed.agent.name.clone(),
        version: parsed.agent.version.clone(),
        sha: parsed.sha,
        file_content: parsed.agent,
        raw_content: parsed.raw_content,
        format: parsed.format,
        promoted: false,
        is_champion: false,
        changelog: None,
        parent_sha: None,
        created_at: now,
        updated_at: now,
    }
}

fn parse_to_value(content: &str, format: &AgentFileFormat) -> Result<serde_json::Value> {
    let trimmed = content.trim();

    match format {
        AgentFileFormat::OpenaiJson | AgentFileFormat::AnthropicJson => {
            serde_json::from_str(trimmed).map_err(|e| AgentForgeError::ParseError(e.to_string()))
        }
        AgentFileFormat::NativeYaml
        | AgentFileFormat::LangchainYaml
        | AgentFileFormat::CrewaiYaml => {
            // Handle Markdown frontmatter
            let yaml_content = if trimmed.starts_with("---") {
                extract_frontmatter(trimmed)?
            } else {
                trimmed.to_string()
            };
            serde_yaml::from_str(&yaml_content)
                .map_err(|e| AgentForgeError::ParseError(e.to_string()))
        }
        AgentFileFormat::CopilotAgentMd => {
            // Parse only the YAML frontmatter as the value;
            // the Markdown body is extracted later in normalize() via the raw content.
            let frontmatter = extract_frontmatter(trimmed)?;
            serde_yaml::from_str(&frontmatter)
                .map_err(|e| AgentForgeError::ParseError(e.to_string()))
        }
    }
}

fn extract_frontmatter(content: &str) -> Result<String> {
    let parts: Vec<&str> = content.splitn(3, "---").collect();
    if parts.len() < 3 {
        return Err(AgentForgeError::ParseError(
            "Malformed Markdown frontmatter: missing closing ---".to_string(),
        ));
    }
    Ok(parts[1].to_string())
}

/// Compute SHA-256 of the file content for content addressing.
pub fn compute_sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    const NATIVE_YAML: &str = indoc! {r#"
        agentforge_schema_version: "1"
        name: customer-support-agent
        version: "1.0.0"
        model:
          provider: openai
          model_id: gpt-4o
          temperature: 0.2
          max_tokens: 2048
        system_prompt: |
          You are a helpful customer support agent.
          Never share pricing without verifying entitlement first.
        tools:
          - name: get_order_status
            description: "Retrieve status of a customer order by order ID."
            parameters:
              type: object
              properties:
                order_id:
                  type: string
              required: [order_id]
        output_schema:
          type: object
          properties:
            response:
              type: string
            action_taken:
              type: string
          required: [response]
        constraints:
          - "Never mention competitor products."
          - "Always confirm order ID before calling get_order_status."
        eval_hints:
          domain: customer_support
          typical_turns: 3
          critical_tools: [get_order_status]
          pass_threshold: 0.85
          scenario_count: 200
    "#};

    #[test]
    fn parses_native_yaml() {
        let result = parse_agent_file(NATIVE_YAML).unwrap();
        assert_eq!(result.agent.name, "customer-support-agent");
        assert_eq!(result.agent.model.model_id, "gpt-4o");
        assert_eq!(result.agent.tools.len(), 1);
        assert_eq!(result.agent.constraints.len(), 2);
        assert_eq!(result.format, AgentFileFormat::NativeYaml);
        assert!(!result.sha.is_empty());
    }

    #[test]
    fn sha_is_deterministic() {
        let a = compute_sha256("hello");
        let b = compute_sha256("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn sha_differs_for_different_content() {
        let a = compute_sha256("hello");
        let b = compute_sha256("world");
        assert_ne!(a, b);
    }

    #[test]
    fn parses_openai_json() {
        let content = r#"{
            "name": "support-bot",
            "instructions": "You are a helpful support agent.",
            "model": "gpt-4o",
            "tools": []
        }"#;
        let result = parse_agent_file(content).unwrap();
        assert_eq!(result.format, AgentFileFormat::OpenaiJson);
        assert_eq!(
            result.agent.system_prompt,
            "You are a helpful support agent."
        );
    }

    #[test]
    fn to_agent_version_sets_fields() {
        let parsed = parse_agent_file(NATIVE_YAML).unwrap();
        let version = to_agent_version(parsed);
        assert_eq!(version.name, "customer-support-agent");
        assert!(!version.promoted);
        assert!(!version.is_champion);
    }
}
