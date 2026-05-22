use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The AgentForge native agent file schema (v1).
/// Also the normalized representation after parsing any supported format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentFile {
    pub agentforge_schema_version: String,
    pub name: String,
    pub version: String,
    pub model: ModelConfig,
    pub system_prompt: String,
    pub tools: Vec<ToolDefinition>,
    pub output_schema: Option<serde_json::Value>,
    pub constraints: Vec<String>,
    pub eval_hints: Option<EvalHints>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub provider: ModelProvider,
    pub model_id: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelProvider {
    Openai,
    Anthropic,
    Ollama,
    Bedrock,
    NvidiaNim,
    Custom,
}

impl std::fmt::Display for ModelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelProvider::Openai => write!(f, "openai"),
            ModelProvider::Anthropic => write!(f, "anthropic"),
            ModelProvider::Ollama => write!(f, "ollama"),
            ModelProvider::Bedrock => write!(f, "bedrock"),
            ModelProvider::NvidiaNim => write!(f, "nvidia_nim"),
            ModelProvider::Custom => write!(f, "custom"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalHints {
    pub domain: Option<String>,
    pub typical_turns: Option<u32>,
    pub critical_tools: Vec<String>,
    pub pass_threshold: Option<f64>,
    pub scenario_count: Option<u32>,
}

impl Default for EvalHints {
    fn default() -> Self {
        Self {
            domain: None,
            typical_turns: Some(3),
            critical_tools: vec![],
            pass_threshold: Some(0.85),
            scenario_count: Some(100),
        }
    }
}

/// Parsed and versioned agent file stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentVersion {
    pub id: uuid::Uuid,
    pub name: String,
    pub version: String,
    pub sha: String,
    pub file_content: AgentFile,
    pub raw_content: String,
    pub format: AgentFileFormat,
    pub promoted: bool,
    pub is_champion: bool,
    pub changelog: Option<String>,
    pub parent_sha: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Supported agent file input formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AgentFileFormat {
    NativeYaml,
    OpenaiJson,
    AnthropicJson,
    LangchainYaml,
    CrewaiYaml,
    /// GitHub Copilot `.agent.md` format — YAML frontmatter + Markdown system prompt body.
    CopilotAgentMd,
}

impl std::fmt::Display for AgentFileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentFileFormat::NativeYaml => write!(f, "native_yaml"),
            AgentFileFormat::OpenaiJson => write!(f, "openai_json"),
            AgentFileFormat::AnthropicJson => write!(f, "anthropic_json"),
            AgentFileFormat::LangchainYaml => write!(f, "langchain_yaml"),
            AgentFileFormat::CrewaiYaml => write!(f, "crewai_yaml"),
            AgentFileFormat::CopilotAgentMd => write!(f, "copilot_agent_md"),
        }
    }
}

impl std::str::FromStr for AgentFileFormat {
    type Err = crate::AgentForgeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "native_yaml" => Ok(AgentFileFormat::NativeYaml),
            "openai_json" => Ok(AgentFileFormat::OpenaiJson),
            "anthropic_json" => Ok(AgentFileFormat::AnthropicJson),
            "langchain_yaml" => Ok(AgentFileFormat::LangchainYaml),
            "crewai_yaml" => Ok(AgentFileFormat::CrewaiYaml),
            "copilot_agent_md" => Ok(AgentFileFormat::CopilotAgentMd),
            _ => Err(crate::AgentForgeError::InvalidFormat(s.to_string())),
        }
    }
}

/// Lint error surfaced during agent file validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintError {
    pub field: String,
    pub message: String,
    pub severity: LintSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LintSeverity {
    Error,
    Warning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_provider_display() {
        assert_eq!(ModelProvider::Openai.to_string(), "openai");
        assert_eq!(ModelProvider::Anthropic.to_string(), "anthropic");
    }

    #[test]
    fn agent_file_format_roundtrip() {
        use std::str::FromStr;
        assert_eq!(
            AgentFileFormat::from_str("native_yaml").unwrap(),
            AgentFileFormat::NativeYaml
        );
        assert_eq!(AgentFileFormat::NativeYaml.to_string(), "native_yaml");
    }

    #[test]
    fn eval_hints_default() {
        let hints = EvalHints::default();
        assert_eq!(hints.pass_threshold, Some(0.85));
        assert_eq!(hints.scenario_count, Some(100));
    }

    // ── 9 new tests ──────────────────────────────────────────────────────────

    #[test]
    fn model_provider_display_all_variants() {
        assert_eq!(ModelProvider::Openai.to_string(), "openai");
        assert_eq!(ModelProvider::Anthropic.to_string(), "anthropic");
        assert_eq!(ModelProvider::Ollama.to_string(), "ollama");
        assert_eq!(ModelProvider::Bedrock.to_string(), "bedrock");
        assert_eq!(ModelProvider::NvidiaNim.to_string(), "nvidia_nim");
        assert_eq!(ModelProvider::Custom.to_string(), "custom");
    }

    #[test]
    fn model_provider_serde_roundtrip() {
        let json = serde_json::to_string(&ModelProvider::NvidiaNim).unwrap();
        assert_eq!(json, r#""nvidia_nim""#);
        let back: ModelProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ModelProvider::NvidiaNim);
    }

    #[test]
    fn agent_file_format_display_all_variants() {
        assert_eq!(AgentFileFormat::NativeYaml.to_string(), "native_yaml");
        assert_eq!(AgentFileFormat::OpenaiJson.to_string(), "openai_json");
        assert_eq!(AgentFileFormat::AnthropicJson.to_string(), "anthropic_json");
        assert_eq!(AgentFileFormat::LangchainYaml.to_string(), "langchain_yaml");
        assert_eq!(AgentFileFormat::CrewaiYaml.to_string(), "crewai_yaml");
        assert_eq!(
            AgentFileFormat::CopilotAgentMd.to_string(),
            "copilot_agent_md"
        );
    }

    #[test]
    fn agent_file_format_from_str_all_variants() {
        use std::str::FromStr;
        let pairs = [
            ("native_yaml", AgentFileFormat::NativeYaml),
            ("openai_json", AgentFileFormat::OpenaiJson),
            ("anthropic_json", AgentFileFormat::AnthropicJson),
            ("langchain_yaml", AgentFileFormat::LangchainYaml),
            ("crewai_yaml", AgentFileFormat::CrewaiYaml),
            ("copilot_agent_md", AgentFileFormat::CopilotAgentMd),
        ];
        for (s, expected) in &pairs {
            assert_eq!(AgentFileFormat::from_str(s).unwrap(), *expected);
        }
    }

    #[test]
    fn agent_file_format_from_str_unknown_returns_err() {
        use std::str::FromStr;
        assert!(AgentFileFormat::from_str("unknown_format").is_err());
    }

    #[test]
    fn eval_hints_default_typical_turns() {
        let hints = EvalHints::default();
        assert_eq!(hints.typical_turns, Some(3));
        assert!(hints.critical_tools.is_empty());
        assert!(hints.domain.is_none());
    }

    #[test]
    fn lint_severity_serde() {
        let json = serde_json::to_string(&LintSeverity::Error).unwrap();
        assert_eq!(json, r#""error""#);
        let back: LintSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(back, LintSeverity::Error);

        let json2 = serde_json::to_string(&LintSeverity::Warning).unwrap();
        assert_eq!(json2, r#""warning""#);
    }

    #[test]
    fn tool_definition_stores_fields() {
        let tool = ToolDefinition {
            name: "get_order".to_string(),
            description: "Fetch an order by ID".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"id": {"type": "string"}}}),
        };
        assert_eq!(tool.name, "get_order");
        assert_eq!(tool.parameters["type"], "object");
    }

    #[test]
    fn agent_version_parent_sha_is_optional() {
        let v = AgentVersion {
            id: uuid::Uuid::new_v4(),
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            sha: "abc123".to_string(),
            file_content: AgentFile {
                agentforge_schema_version: "1".to_string(),
                name: "test".to_string(),
                version: "1.0.0".to_string(),
                model: ModelConfig {
                    provider: ModelProvider::Openai,
                    model_id: "gpt-4o".to_string(),
                    temperature: None,
                    max_tokens: None,
                    top_p: None,
                },
                system_prompt: "You are helpful.".to_string(),
                tools: vec![],
                output_schema: None,
                constraints: vec![],
                eval_hints: None,
                metadata: None,
            },
            raw_content: "{}".to_string(),
            format: AgentFileFormat::NativeYaml,
            promoted: false,
            is_champion: false,
            changelog: None,
            parent_sha: Some("parent_sha_123".to_string()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        assert_eq!(v.parent_sha.as_deref(), Some("parent_sha_123"));
    }
}
