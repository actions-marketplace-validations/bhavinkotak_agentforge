use crate::optimizer::OptimizerConfig;
use agentforge_core::{AgentFile, AgentForgeError, Result, Scorecard, Trace};

/// Rewrite the system prompt using LLM to improve clarity and constraint specificity.
pub async fn rewrite_prompt(
    agent: &AgentFile,
    scorecard: &Scorecard,
    config: &OptimizerConfig,
    n: usize,
) -> Result<Vec<AgentFile>> {
    if config.llm_api_key.is_empty() {
        return Err(AgentForgeError::ConfigError(
            "No LLM API key configured for prompt rewriting".to_string(),
        ));
    }

    let failure_summary = format!(
        "Task completion: {:.0}%, Tool selection: {:.0}%, Arg correctness: {:.0}%, Instruction adherence: {:.0}%",
        scorecard.dimension_scores.task_completion * 100.0,
        scorecard.dimension_scores.tool_selection * 100.0,
        scorecard.dimension_scores.argument_correctness * 100.0,
        scorecard.dimension_scores.instruction_adherence * 100.0,
    );

    let prompt = format!(
        r#"You are an expert AI prompt engineer. Rewrite the following agent system prompt to improve its performance.

Current performance issues:
{failure_summary}

Original system prompt:
---
{system_prompt}
---

Current constraints:
{constraints}

Generate {n} improved versions of the system prompt. Each version should:
1. Be more specific and unambiguous
2. Make constraints clearer and harder to violate
3. Provide better guidance on when and how to use tools
4. Address the specific performance gaps above
5. Keep the core purpose and tone intact

Respond with a JSON array of {n} strings, each being a complete improved system prompt."#,
        failure_summary = failure_summary,
        system_prompt = agent.system_prompt,
        constraints = agent.constraints.join("\n"),
        n = n,
    );

    let response = call_llm_api(&prompt, config, n).await?;

    parse_prompt_variants(&response, agent, n)
}

fn parse_prompt_variants(
    response: &serde_json::Value,
    base_agent: &AgentFile,
    n: usize,
) -> Result<Vec<AgentFile>> {
    // Try to find an array of strings in the response — the LLM may use any key name
    let arr = if let Some(arr) = response.as_array() {
        arr.to_vec()
    } else if let Some(obj) = response.as_object() {
        // Scan all keys and pick the first that holds a non-empty array
        obj.values()
            .find_map(|v| {
                let a = v.as_array()?;
                if a.is_empty() {
                    None
                } else {
                    Some(a.to_vec())
                }
            })
            .ok_or_else(|| {
                AgentForgeError::ParseError(
                    "LLM did not return a valid array of prompts".to_string(),
                )
            })?
    } else {
        return Err(AgentForgeError::ParseError(
            "LLM did not return a valid array of prompts".to_string(),
        ));
    };

    let variants: Vec<AgentFile> = arr
        .iter()
        .take(n)
        .filter_map(|v| v.as_str())
        .map(|prompt_text| {
            let mut new_agent = base_agent.clone();
            new_agent.system_prompt = prompt_text.to_string();
            // Bump patch version
            new_agent.version = bump_patch_version(&new_agent.version);
            new_agent
        })
        .collect();

    if variants.is_empty() {
        return Err(AgentForgeError::OptimizationError(
            "LLM returned no usable prompt variants".to_string(),
        ));
    }

    Ok(variants)
}

/// Rewrite tool descriptions to include examples, type constraints, and misuse warnings.
pub async fn rewrite_tool_descriptions(
    agent: &AgentFile,
    scorecard: &Scorecard,
    config: &OptimizerConfig,
    n: usize,
) -> Result<Vec<AgentFile>> {
    if config.llm_api_key.is_empty() {
        return Err(AgentForgeError::ConfigError(
            "No LLM API key configured".to_string(),
        ));
    }

    let tools_json = serde_json::to_string_pretty(&agent.tools)
        .map_err(|e| AgentForgeError::SerializationError(e.to_string()))?;

    let prompt = format!(
        r#"You are an expert at writing clear AI tool definitions. Improve the following tool descriptions.

Current tool selection accuracy: {:.0}%
Current argument correctness: {:.0}%

Current tools:
{tools_json}

Generate {n} improved versions of the tools array. Each improvement should:
1. Add concrete examples of valid parameter values
2. Specify exact format requirements (e.g., "ORD-XXXXXXXX" not just "string")
3. Add warnings about common misuse patterns
4. Clarify relationships between parameters
5. Keep the tool names and structure intact

Respond with a JSON array of {n} items. Each item is a tools array (same structure as input)."#,
        scorecard.dimension_scores.tool_selection * 100.0,
        scorecard.dimension_scores.argument_correctness * 100.0,
        tools_json = tools_json,
        n = n,
    );

    let response = call_llm_api(&prompt, config, n).await?;

    parse_tool_variants(&response, agent, n)
}

fn parse_tool_variants(
    response: &serde_json::Value,
    base_agent: &AgentFile,
    n: usize,
) -> Result<Vec<AgentFile>> {
    // Scan any array field — the LLM may use any key name (e.g. "tools", "variants", "improved_tools")
    let arr = if let Some(arr) = response.as_array() {
        arr.to_vec()
    } else if let Some(obj) = response.as_object() {
        obj.values()
            .find_map(|v| {
                let a = v.as_array()?;
                if a.is_empty() {
                    None
                } else {
                    Some(a.to_vec())
                }
            })
            .ok_or_else(|| {
                AgentForgeError::ParseError(
                    "LLM did not return a valid array of tool variants".to_string(),
                )
            })?
    } else {
        return Err(AgentForgeError::ParseError(
            "LLM did not return a valid array of tool variants".to_string(),
        ));
    };

    let variants: Vec<AgentFile> = arr
        .iter()
        .take(n)
        .filter_map(|v| {
            let tools = v.as_array()?;
            let parsed_tools: Vec<agentforge_core::ToolDefinition> = tools
                .iter()
                .filter_map(|t| serde_json::from_value(t.clone()).ok())
                .collect();
            if parsed_tools.is_empty() {
                return None;
            }
            let mut new_agent = base_agent.clone();
            new_agent.tools = parsed_tools;
            new_agent.version = bump_patch_version(&new_agent.version);
            Some(new_agent)
        })
        .collect();

    if variants.is_empty() {
        return Err(AgentForgeError::OptimizationError(
            "No valid tool variants produced".to_string(),
        ));
    }

    Ok(variants)
}

/// Tighten the output schema by marking more fields as required and adding enum constraints.
pub fn tighten_output_schema(agent: &AgentFile) -> Option<AgentFile> {
    let schema = agent.output_schema.as_ref()?;
    let properties = schema.get("properties")?.as_object()?;

    let mut new_schema = schema.clone();
    let mut all_fields: Vec<String> = properties.keys().cloned().collect();
    all_fields.sort();

    // Mark all fields as required (not just some)
    new_schema["required"] = serde_json::json!(all_fields);

    // Add additionalProperties: false to prevent extra fields
    new_schema["additionalProperties"] = serde_json::json!(false);

    let mut new_agent = agent.clone();
    new_agent.output_schema = Some(new_schema);
    new_agent.version = bump_patch_version(&new_agent.version);
    Some(new_agent)
}

/// Inject few-shot examples from top-scoring passing traces into the system prompt.
pub fn inject_few_shot_examples(agent: &AgentFile, passing_traces: &[Trace]) -> Result<AgentFile> {
    if passing_traces.is_empty() {
        return Err(AgentForgeError::OptimizationError(
            "No passing traces available for few-shot injection".to_string(),
        ));
    }

    // Select the top 5 traces by aggregate score
    let mut scored: Vec<(f64, &Trace)> = passing_traces
        .iter()
        .filter_map(|t| t.aggregate_score.map(|s| (s, t)))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(5);

    let examples: Vec<String> = scored
        .iter()
        .filter_map(|(_, trace)| {
            let output = trace.final_output.as_ref()?;
            let response = output.get("response")?.as_str()?;
            Some(format!(
                "Example response:\n{}",
                &response[..response.len().min(300)]
            ))
        })
        .collect();

    if examples.is_empty() {
        return Err(AgentForgeError::OptimizationError(
            "No usable examples extracted from traces".to_string(),
        ));
    }

    let examples_section = format!(
        "\n\n## Examples of Excellent Responses\n\n{}",
        examples.join("\n\n---\n\n")
    );

    let mut new_agent = agent.clone();
    new_agent.system_prompt = format!("{}{}", agent.system_prompt, examples_section);
    new_agent.version = bump_patch_version(&new_agent.version);

    Ok(new_agent)
}

async fn call_llm_api(
    prompt: &str,
    config: &OptimizerConfig,
    _n: usize,
) -> Result<serde_json::Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

    let body = serde_json::json!({
        "model": config.llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert AI agent optimizer. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "n": 1
    });

    // Retry up to 3 times with exponential backoff (handles post-eval rate-limit recovery)
    let mut last_err = String::new();
    for attempt in 0..3u32 {
        if attempt > 0 {
            let delay_secs = 10u64 * (1 << (attempt - 1)); // 10s, 20s
            tracing::info!(
                attempt,
                delay_secs,
                "Retrying optimizer LLM call after delay"
            );
            tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
        }

        let send_result = client
            .post(format!("{}/chat/completions", config.llm_base_url))
            .header("Authorization", format!("Bearer {}", config.llm_api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await;

        let resp = match send_result {
            Ok(r) => r,
            Err(e) => {
                last_err = format!("connection error: {e}");
                tracing::warn!(attempt, error = %e, "Optimizer LLM connection failed, will retry");
                continue;
            }
        };

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            last_err = "rate limited (429)".to_string();
            tracing::warn!(attempt, "Optimizer LLM rate-limited, will retry");
            continue;
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AgentForgeError::LlmError {
                provider: "optimizer".to_string(),
                message: format!("HTTP {status}: {text}"),
            });
        }

        let raw: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

        let content = raw["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[]");

        return serde_json::from_str(content)
            .map_err(|e| AgentForgeError::ParseError(format!("LLM returned invalid JSON: {e}")));
    }

    Err(AgentForgeError::LlmError {
        provider: "optimizer".to_string(),
        message: format!("LLM call failed after 3 attempts: {last_err}"),
    })
}

/// Bump the patch component of a semver string (e.g. "1.0.0" → "1.0.1").
/// Exposed for use by other modules in this crate.
pub fn bump_patch_version_pub(version: &str) -> String {
    bump_patch_version(version)
}

fn bump_patch_version(version: &str) -> String {
    // Parse semver and bump patch
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() == 3 {
        if let Ok(patch) = parts[2].parse::<u32>() {
            return format!("{}.{}.{}", parts[0], parts[1], patch + 1);
        }
    }
    format!("{}-opt", version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_core::{ModelConfig, ModelProvider, ToolDefinition};

    fn make_agent_with_schema() -> AgentFile {
        AgentFile {
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
            tools: vec![ToolDefinition {
                name: "search".to_string(),
                description: "Search".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
            }],
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "action": {"type": "string"}
                },
                "required": ["response"]
            })),
            constraints: vec!["Never share passwords.".to_string()],
            eval_hints: None,
            metadata: None,
        }
    }

    #[test]
    fn tighten_schema_marks_all_required() {
        let agent = make_agent_with_schema();
        let variant = tighten_output_schema(&agent).unwrap();
        let required = variant.output_schema.as_ref().unwrap()["required"]
            .as_array()
            .unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("response")));
        assert!(required.iter().any(|v| v.as_str() == Some("action")));
        // additional_properties should be false
        assert_eq!(
            variant.output_schema.as_ref().unwrap()["additionalProperties"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn tighten_schema_returns_none_when_no_schema() {
        let mut agent = make_agent_with_schema();
        agent.output_schema = None;
        assert!(tighten_output_schema(&agent).is_none());
    }

    #[test]
    fn bump_patch_version_works() {
        assert_eq!(bump_patch_version("1.0.0"), "1.0.1");
        assert_eq!(bump_patch_version("2.3.7"), "2.3.8");
        assert_eq!(bump_patch_version("invalid"), "invalid-opt");
    }

    #[test]
    fn inject_few_shot_returns_err_on_empty() {
        let agent = make_agent_with_schema();
        assert!(inject_few_shot_examples(&agent, &[]).is_err());
    }
}
