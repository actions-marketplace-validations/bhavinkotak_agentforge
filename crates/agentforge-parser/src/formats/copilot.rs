use agentforge_core::{AgentFile, EvalHints, ModelConfig, ModelProvider, Result, ToolDefinition};
use std::collections::HashMap;

/// Normalize a GitHub Copilot `.agent.md` file into `AgentFile`.
///
/// Format: YAML frontmatter (`---`) followed by Markdown body.
/// Frontmatter fields:
///   - `name`         — agent display name (required)
///   - `description`  — short description (optional)
///   - `model`        — LLM model string, e.g. "GPT-4.1", "claude-sonnet-4-5" (optional)
///   - `tools`        — list of capability strings like "read", "github/*" (optional)
///
/// The Markdown body after the frontmatter becomes `system_prompt`.
pub fn normalize(frontmatter: &serde_json::Value, system_prompt_body: &str) -> Result<AgentFile> {
    let name = frontmatter
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("copilot-agent")
        .to_string();

    let description = frontmatter
        .get("description")
        .and_then(|v| v.as_str())
        .map(String::from);

    // System prompt is the full Markdown body
    let system_prompt = system_prompt_body.trim().to_string();

    // Model: infer provider from model string
    let model = parse_model(frontmatter);

    // Tools: Copilot tools are capability reference strings (e.g. "github/*", "read"),
    // not structured tool definitions. We store them as minimal ToolDefinitions so they
    // appear in the agent representation and can be used in scenario generation.
    let tools = parse_copilot_tools(frontmatter);

    // Build metadata preserving Copilot-specific fields
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    if let Some(desc) = &description {
        metadata.insert(
            "description".to_string(),
            serde_json::Value::String(desc.clone()),
        );
    }
    if let Some(arg_hint) = frontmatter.get("argument-hint").and_then(|v| v.as_str()) {
        metadata.insert(
            "argument_hint".to_string(),
            serde_json::Value::String(arg_hint.to_string()),
        );
    }
    if let Some(handoffs) = frontmatter.get("handoffs") {
        metadata.insert("handoffs".to_string(), handoffs.clone());
    }
    if let Some(mcp_servers) = frontmatter.get("mcp-servers") {
        metadata.insert("mcp_servers".to_string(), mcp_servers.clone());
    }

    Ok(AgentFile {
        agentforge_schema_version: "1".to_string(),
        name,
        version: "1.0.0".to_string(),
        model,
        system_prompt,
        tools,
        output_schema: None,
        constraints: vec![],
        eval_hints: Some(EvalHints::default()),
        metadata: if metadata.is_empty() {
            None
        } else {
            Some(metadata)
        },
    })
}

/// Parse the `model` frontmatter field into a `ModelConfig`.
/// Supports model strings like "GPT-4.1", "gpt-4o", "claude-sonnet-4-5", "o3", etc.
fn parse_model(frontmatter: &serde_json::Value) -> ModelConfig {
    let model_str = frontmatter
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-4o");

    let lower = model_str.to_lowercase();

    let (provider, model_id) = if lower.contains("claude") || lower.contains("anthropic") {
        (ModelProvider::Anthropic, model_str.to_string())
    } else if lower.contains("ollama") || lower.starts_with("ollama/") {
        let id = model_str.strip_prefix("ollama/").unwrap_or(model_str);
        (ModelProvider::Ollama, id.to_string())
    } else {
        // Default: OpenAI (covers gpt-*, o1, o3, GPT-4.1, etc.)
        (ModelProvider::Openai, model_str.to_string())
    };

    ModelConfig {
        provider,
        model_id,
        temperature: None,
        max_tokens: None,
        top_p: None,
    }
}

/// Parse Copilot tool capability strings into minimal `ToolDefinition` entries.
///
/// Copilot tools are references like `"read"`, `"github/*"`, `"context7/*"`, not full schemas.
/// We create lightweight ToolDefinitions so the rest of AgentForge can reason about them.
fn parse_copilot_tools(frontmatter: &serde_json::Value) -> Vec<ToolDefinition> {
    let tools_val = match frontmatter.get("tools") {
        Some(t) => t,
        None => return vec![],
    };

    let tool_refs: Vec<String> = match tools_val {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        serde_json::Value::String(s) => vec![s.clone()],
        _ => return vec![],
    };

    tool_refs
        .into_iter()
        .map(|capability| {
            // Derive a human-readable name: "github/*" → "github", "read/readFile" → "readFile"
            let display_name = capability
                .rsplit('/')
                .next()
                .map(|s| {
                    if s == "*" {
                        capability
                            .split('/')
                            .next()
                            .unwrap_or(&capability)
                            .to_string()
                    } else {
                        s.to_string()
                    }
                })
                .unwrap_or_else(|| capability.clone());

            let (description, parameters) = capability_schema(&capability, &display_name);

            ToolDefinition {
                name: display_name,
                description,
                parameters,
            }
        })
        .collect()
}

/// Return a (description, parameters JSON schema) pair for a Copilot capability string.
///
/// Provides meaningful parameter schemas so LLMs can actually call these tools during
/// eval scenarios, rather than refusing because `properties: {}` signals no parameters.
fn capability_schema(capability: &str, display_name: &str) -> (String, serde_json::Value) {
    // Normalise to lowercase leaf for pattern matching
    let leaf = display_name.to_lowercase();

    match leaf.as_str() {
        // GitHub API — flexible query/action parameter
        "github" => (
            "Interact with GitHub: search repositories, list files, read file contents, \
             search code, list issues, pull requests, workflows, and other GitHub API operations."
                .to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The GitHub operation or search query to perform \
                                        (e.g. 'list workflow files in .github/workflows/', \
                                        'search code for TODO', 'get file contents of README.md')."
                    }
                },
                "required": ["query"],
                "x-copilot-capability": capability
            }),
        ),

        // File search
        "filesearch" | "file_search" => (
            "Search the repository for files matching a name pattern or glob.".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Filename pattern, glob, or partial path to search for \
                                        (e.g. '*.yml', '.github/workflows/*.yml', 'Dockerfile')."
                    }
                },
                "required": ["query"],
                "x-copilot-capability": capability
            }),
        ),

        // Codebase / semantic search
        "codebase" | "search_codebase" | "searchcodebase" => (
            "Semantically search the codebase for relevant code, functions, or patterns."
                .to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language or keyword search query to find \
                                        relevant code in the workspace."
                    }
                },
                "required": ["query"],
                "x-copilot-capability": capability
            }),
        ),

        // Read file
        "readfile" | "read_file" => (
            "Read the contents of a file in the repository.".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file to read \
                                        (e.g. '.github/workflows/ci.yml')."
                    }
                },
                "required": ["file_path"],
                "x-copilot-capability": capability
            }),
        ),

        // Edit / write files
        "editfiles" | "edit_files" => (
            "Create or edit one or more files in the repository.".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file to create or modify."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new content to write to the file."
                    }
                },
                "required": ["file_path", "content"],
                "x-copilot-capability": capability
            }),
        ),

        // Run terminal command
        "runinterminal" | "run_in_terminal" | "terminal" => (
            "Execute a shell command in the project terminal.".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run (e.g. 'actionlint .github/workflows/ci.yml')."
                    }
                },
                "required": ["command"],
                "x-copilot-capability": capability
            }),
        ),

        // Fallback: generic single-input tool
        _ => (
            format!("Copilot capability: {capability}"),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": format!("Input for the {display_name} capability.")
                    }
                },
                "required": ["input"],
                "x-copilot-capability": capability
            }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_basic_copilot_agent() {
        let frontmatter = serde_json::json!({
            "name": "GitHub Actions Expert",
            "description": "Specialist in secure CI/CD workflows",
            "model": "GPT-4.1",
            "tools": ["github/*", "search/codebase", "edit/editFiles"]
        });
        let body = "# GitHub Actions Expert\n\nYou help teams build secure workflows.";

        let agent = normalize(&frontmatter, body).unwrap();

        assert_eq!(agent.name, "GitHub Actions Expert");
        assert_eq!(agent.model.model_id, "GPT-4.1");
        assert_eq!(agent.model.provider, ModelProvider::Openai);
        assert!(agent.system_prompt.contains("GitHub Actions Expert"));
        assert_eq!(agent.tools.len(), 3);
        assert_eq!(agent.tools[0].name, "github");
        assert_eq!(agent.tools[1].name, "codebase");
        assert_eq!(agent.tools[2].name, "editFiles");
        // Each tool must have at least one required parameter so models can call them
        for tool in &agent.tools {
            let props = tool
                .parameters
                .get("properties")
                .and_then(|p| p.as_object());
            assert!(
                props.map(|p| !p.is_empty()).unwrap_or(false),
                "Tool '{}' must have non-empty properties",
                tool.name
            );
        }
    }

    #[test]
    fn normalizes_claude_model() {
        let frontmatter = serde_json::json!({
            "name": "Claude Agent",
            "model": "claude-sonnet-4-5"
        });
        let agent = normalize(&frontmatter, "You are helpful.").unwrap();
        assert_eq!(agent.model.provider, ModelProvider::Anthropic);
        assert_eq!(agent.model.model_id, "claude-sonnet-4-5");
    }

    #[test]
    fn defaults_model_when_absent() {
        let frontmatter = serde_json::json!({ "name": "No Model Agent" });
        let agent = normalize(&frontmatter, "Do stuff.").unwrap();
        assert_eq!(agent.model.model_id, "gpt-4o");
        assert_eq!(agent.model.provider, ModelProvider::Openai);
    }

    #[test]
    fn stores_description_in_metadata() {
        let frontmatter = serde_json::json!({
            "name": "Test",
            "description": "A helpful test agent"
        });
        let agent = normalize(&frontmatter, "System prompt.").unwrap();
        let meta = agent.metadata.unwrap();
        assert_eq!(
            meta["description"],
            serde_json::Value::String("A helpful test agent".to_string())
        );
    }

    #[test]
    fn empty_tools_yields_no_tool_definitions() {
        let frontmatter = serde_json::json!({ "name": "No Tools" });
        let agent = normalize(&frontmatter, "Prompt.").unwrap();
        assert!(agent.tools.is_empty());
    }

    /// Regression: every known Copilot capability must produce a non-empty
    /// `properties` map so models can call the tool with real arguments.
    /// Previously all tools had `properties: {}` which caused models to refuse
    /// to invoke them (no parameters = no-op tool).
    #[test]
    fn all_known_capabilities_have_non_empty_schemas() {
        let capabilities = [
            "github/*",
            "search/fileSearch",
            "search/codebase",
            "read/readFile",
            "edit/editFiles",
            "execute/runInTerminal",
        ];
        let frontmatter = serde_json::json!({
            "name": "Full Agent",
            "tools": capabilities
        });
        let agent = normalize(&frontmatter, "Prompt.").unwrap();
        assert_eq!(agent.tools.len(), capabilities.len());
        for tool in &agent.tools {
            let props = tool
                .parameters
                .get("properties")
                .and_then(|p| p.as_object())
                .expect(&format!("'{}' must have a properties object", tool.name));
            assert!(
                !props.is_empty(),
                "Tool '{}' has empty properties — models cannot call it",
                tool.name
            );
            let required = tool
                .parameters
                .get("required")
                .and_then(|r| r.as_array())
                .expect(&format!("'{}' must have a required array", tool.name));
            assert!(
                !required.is_empty(),
                "Tool '{}' has no required fields — models may skip it",
                tool.name
            );
        }
    }

    /// Unknown/custom capabilities should fall back to a generic `input` field,
    /// not an empty schema.
    #[test]
    fn unknown_capability_falls_back_to_generic_schema() {
        let frontmatter = serde_json::json!({
            "name": "Custom Agent",
            "tools": ["custom/myTool", "context7/*"]
        });
        let agent = normalize(&frontmatter, "Prompt.").unwrap();
        for tool in &agent.tools {
            let props = tool
                .parameters
                .get("properties")
                .and_then(|p| p.as_object())
                .expect(&format!("'{}' must have properties", tool.name));
            assert!(!props.is_empty(), "Fallback tool '{}' must not have empty properties", tool.name);
        }
    }
}
