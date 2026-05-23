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

    // Truncate system prompt to avoid overwhelming a small local model.
    // We pass the first 600 chars for context, then ask for concise focused improvements.
    let prompt_excerpt = if agent.system_prompt.len() > 600 {
        format!(
            "{}... [prompt continues]",
            &agent.system_prompt[..600]
        )
    } else {
        agent.system_prompt.clone()
    };

    let prompt = format!(
        r#"You are an expert AI prompt engineer. Improve the agent system prompt below.

Current performance issues:
{failure_summary}

Beginning of system prompt (first 600 chars):
---
{prompt_excerpt}
---

Generate {n} concise improved system prompts (max 150 words each) that:
1. Address the specific performance gaps above
2. Keep the same role and purpose
3. Add clearer instructions for common failure cases

Respond with a JSON object with key "variants" containing an array of {n} short improved system prompt strings. Example format: {{"variants": ["improved prompt 1", "improved prompt 2"]}}"#,
        failure_summary = failure_summary,
        prompt_excerpt = prompt_excerpt,
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
    // Prefer the "variants" key, then fall back to scanning all keys for a non-empty array
    let arr = if let Some(arr) = response.as_array() {
        arr.to_vec()
    } else if let Some(obj) = response.as_object() {
        if let Some(arr) = obj.get("variants").and_then(|v| v.as_array()) {
            if !arr.is_empty() {
                arr.to_vec()
            } else {
                return Err(AgentForgeError::ParseError(
                    "LLM returned empty variants array".to_string(),
                ));
            }
        } else {
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
        }
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

Respond with a JSON object with key "variants" containing an array of {n} items. Each item is a tools array (same structure as input). Example format: {{"variants": [[{{"name":"tool1","description":"improved","parameters":{{}}}}], [{{"name":"tool1","description":"alt","parameters":{{}}}}]]}}"#,
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
    // Prefer the "variants" key, then fall back to scanning all keys
    let arr = if let Some(arr) = response.as_array() {
        arr.to_vec()
    } else if let Some(obj) = response.as_object() {
        if let Some(arr) = obj.get("variants").and_then(|v| v.as_array()) {
            if !arr.is_empty() {
                arr.to_vec()
            } else {
                return Err(AgentForgeError::ParseError(
                    "LLM returned empty tool variants array".to_string(),
                ));
            }
        } else {
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
        }
    } else {
        return Err(AgentForgeError::ParseError(
            "LLM did not return a valid array of tool variants".to_string(),
        ));
    };

    // Flat fallback: if array elements are tool objects (not arrays-of-tools), treat the
    // whole array as a single tool variant (the LLM returned one improved tool set directly).
    let flat_fallback = arr.first().map(|v| v.is_object()).unwrap_or(false);
    let variants: Vec<AgentFile> = if flat_fallback {
        let parsed_tools: Vec<agentforge_core::ToolDefinition> = arr
            .iter()
            .filter_map(|t| serde_json::from_value(t.clone()).ok())
            .collect();
        if parsed_tools.is_empty() {
            vec![]
        } else {
            let mut new_agent = base_agent.clone();
            new_agent.tools = parsed_tools;
            new_agent.version = bump_patch_version(&new_agent.version);
            vec![new_agent]
        }
    } else {
        arr.iter()
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
            .collect()
    };

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
        "max_tokens": 512,
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

    // ── bump_patch_version_pub ────────────────────────────────────────────────

    #[test]
    fn bump_patch_version_pub_matches_private() {
        // The pub wrapper must return identical results to the private function.
        assert_eq!(bump_patch_version_pub("1.0.0"), "1.0.1");
        assert_eq!(bump_patch_version_pub("3.2.9"), "3.2.10");
        assert_eq!(bump_patch_version_pub("invalid"), "invalid-opt");
    }

    #[test]
    fn bump_patch_version_large_patch_number() {
        assert_eq!(bump_patch_version("1.0.99"), "1.0.100");
        assert_eq!(bump_patch_version("1.0.999"), "1.0.1000");
    }

    #[test]
    fn bump_patch_version_zero_patch() {
        assert_eq!(bump_patch_version("0.0.0"), "0.0.1");
    }

    #[test]
    fn bump_patch_version_does_not_touch_major_minor() {
        let result = bump_patch_version("5.12.3");
        assert!(result.starts_with("5.12."), "major.minor must be preserved");
        assert_eq!(result, "5.12.4");
    }

    #[test]
    fn bump_patch_version_empty_string_returns_fallback() {
        let result = bump_patch_version("");
        assert!(!result.is_empty(), "result must not be empty string");
    }

    #[test]
    fn bump_patch_version_only_two_parts_returns_fallback() {
        // "1.0" has no patch component — should return fallback
        let result = bump_patch_version("1.0");
        assert!(result.ends_with("-opt"), "must use -opt fallback: {result}");
    }

    #[test]
    fn bump_patch_version_pub_zero_to_one() {
        assert_eq!(bump_patch_version_pub("2.5.0"), "2.5.1");
    }

    // ── parse_prompt_variants ─────────────────────────────────────────────────

    #[test]
    fn parse_prompt_variants_from_top_level_array() {
        let agent = make_agent_with_schema();
        let response = serde_json::json!(["prompt A", "prompt B", "prompt C"]);
        let variants = parse_prompt_variants(&response, &agent, 3).unwrap();
        assert_eq!(variants.len(), 3);
        assert_eq!(variants[0].system_prompt, "prompt A");
        assert_eq!(variants[1].system_prompt, "prompt B");
    }

    #[test]
    fn parse_prompt_variants_from_object_key() {
        let agent = make_agent_with_schema();
        let response = serde_json::json!({
            "improved_prompts": ["new prompt 1", "new prompt 2"]
        });
        let variants = parse_prompt_variants(&response, &agent, 2).unwrap();
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0].system_prompt, "new prompt 1");
    }

    #[test]
    fn parse_prompt_variants_caps_at_n() {
        let agent = make_agent_with_schema();
        let response = serde_json::json!(["a", "b", "c", "d", "e"]);
        let variants = parse_prompt_variants(&response, &agent, 2).unwrap();
        assert_eq!(variants.len(), 2);
    }

    #[test]
    fn parse_prompt_variants_bumps_version() {
        let mut agent = make_agent_with_schema();
        agent.version = "1.2.3".to_string();
        let response = serde_json::json!(["new prompt"]);
        let variants = parse_prompt_variants(&response, &agent, 1).unwrap();
        assert_eq!(variants[0].version, "1.2.4");
    }

    #[test]
    fn parse_prompt_variants_error_on_empty_object() {
        let agent = make_agent_with_schema();
        let response = serde_json::json!({});
        assert!(parse_prompt_variants(&response, &agent, 1).is_err());
    }

    #[test]
    fn parse_prompt_variants_error_on_non_array_non_object() {
        let agent = make_agent_with_schema();
        let response = serde_json::json!("just a string");
        assert!(parse_prompt_variants(&response, &agent, 1).is_err());
    }

    #[test]
    fn parse_prompt_variants_filters_non_string_array_items() {
        let agent = make_agent_with_schema();
        let response = serde_json::json!([42, "valid prompt", null]);
        let variants = parse_prompt_variants(&response, &agent, 3).unwrap();
        // Only 1 valid string out of 3
        assert_eq!(variants.len(), 1);
    }

    // ── tighten_output_schema ─────────────────────────────────────────────────

    #[test]
    fn tighten_schema_adds_additional_properties_false() {
        let agent = make_agent_with_schema();
        let variant = tighten_output_schema(&agent).unwrap();
        assert_eq!(
            variant.output_schema.as_ref().unwrap()["additionalProperties"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn tighten_schema_version_bumped() {
        let mut agent = make_agent_with_schema();
        agent.version = "3.0.5".to_string();
        let variant = tighten_output_schema(&agent).unwrap();
        assert_eq!(variant.version, "3.0.6");
    }

    #[test]
    fn tighten_schema_preserves_properties() {
        let agent = make_agent_with_schema();
        let variant = tighten_output_schema(&agent).unwrap();
        let schema = variant.output_schema.as_ref().unwrap();
        // Properties from the original schema must still be present
        assert!(schema["properties"].is_object());
    }

    #[test]
    fn tighten_schema_returns_none_no_schema() {
        let agent = make_agent_with_schema();
        let mut no_schema = agent.clone();
        no_schema.output_schema = None;
        assert!(tighten_output_schema(&no_schema).is_none());
    }

    #[test]
    fn tighten_schema_preserves_agent_name() {
        let agent = make_agent_with_schema();
        let variant = tighten_output_schema(&agent).unwrap();
        assert_eq!(variant.name, agent.name);
    }

    // ── inject_few_shot_examples ──────────────────────────────────────────────

    #[test]
    fn inject_few_shot_ok_with_traces() {
        use agentforge_core::{FailureCluster, Trace, TraceStatus};
        use uuid::Uuid;
        let agent = make_agent_with_schema();
        let traces: Vec<Trace> = (0..3)
            .map(|i| Trace {
                id: Uuid::new_v4(),
                run_id: Uuid::new_v4(),
                scenario_id: Uuid::new_v4(),
                status: TraceStatus::Pass,
                steps: vec![],
                final_output: Some(serde_json::json!({"response": format!("agent answer {i}")})),
                scores: None,
                aggregate_score: Some(0.9),
                failure_cluster: FailureCluster::NoFailure,
                failure_reason: None,
                review_needed: false,
                llm_calls: 1,
                tool_invocations: 0,
                input_tokens: 10,
                output_tokens: 20,
                latency_ms: 100,
                retry_count: 0,
                seed: 0,
                created_at: chrono::Utc::now(),
            })
            .collect();
        let result = inject_few_shot_examples(&agent, &traces);
        assert!(result.is_ok(), "inject_few_shot_examples failed: {:?}", result.err());
        let variant = result.unwrap();
        // System prompt must contain the original content and the examples section
        assert!(
            variant.system_prompt.contains("Examples of Excellent Responses"),
            "Injected prompt must contain the examples section"
        );
    }

    // ── parse_tool_variants ───────────────────────────────────────────────────

    #[test]
    fn parse_tool_variants_from_improved_variants_key() {
        use agentforge_core::ToolDefinition;
        let mut agent = make_agent_with_schema();
        agent.tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search for things".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }];
        // Each element of the outer array is a *complete array of tools* for one variant.
        let response = serde_json::json!({
            "improved_variants": [
                [{"name": "search", "description": "Search the web for information", "parameters": {"type": "object", "properties": {}, "required": []}}]
            ]
        });
        let variants = parse_tool_variants(&response, &agent, 1).unwrap();
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].tools[0].description, "Search the web for information");
    }
}
