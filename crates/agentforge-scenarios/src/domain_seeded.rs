use agentforge_core::{
    AgentFile, AgentForgeError, DifficultyTier, Result, Scenario, ScenarioExpected, ScenarioInput,
    ScenarioSource,
};
use chrono::Utc;
use regex::Regex;
use uuid::Uuid;

pub struct DomainSeededConfig {
    pub count: usize,
    pub agent_id: Uuid,
    pub llm_base_url: Option<String>,
    pub llm_api_key: Option<String>,
    pub llm_model: String,
}

/// Generate domain-relevant test scenarios using an LLM.
/// Falls back to heuristic domain patterns if no LLM is configured.
pub async fn generate_domain_seeded_scenarios(
    agent: &AgentFile,
    config: &DomainSeededConfig,
) -> Result<Vec<Scenario>> {
    let domain = agent
        .eval_hints
        .as_ref()
        .and_then(|h| h.domain.as_deref())
        .unwrap_or("general");

    let keywords = extract_domain_keywords(&agent.system_prompt);

    if config.llm_api_key.is_some() {
        generate_via_llm(agent, config, domain, &keywords).await
    } else {
        generate_heuristic(agent, config, domain, &keywords)
    }
}

async fn generate_via_llm(
    agent: &AgentFile,
    config: &DomainSeededConfig,
    domain: &str,
    keywords: &[String],
) -> Result<Vec<Scenario>> {
    let base_url = config
        .llm_base_url
        .clone()
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

    let api_key = config.llm_api_key.as_deref().unwrap_or("");

    let tool_names: Vec<&str> = agent.tools.iter().map(|t| t.name.as_str()).collect();
    let tool_list = tool_names.join(", ");
    let keyword_list = keywords.join(", ");

    let prompt = format!(
        r#"Generate {count} realistic test scenarios for an AI agent.

Agent: {agent_name}
Domain: {domain}
Keywords: {keyword_list}
Available tools: {tool_list}
System prompt excerpt: {prompt_excerpt}

Rules:
1. Write user_message as a real user would — describe the problem, never name a tool.
2. Make each scenario concrete and specific to the domain (include file paths, tech stack, configs).
3. Vary difficulty: include easy, medium, and at least one hard scenario.
4. expected_tools must only contain names from the available tools list.

Respond with ONLY a JSON array (no markdown, no explanation) where each element has:
  "user_message": string
  "pass_criteria": string describing what a correct response looks like
  "difficulty": "easy" | "medium" | "hard" | "edge"
  "expected_tools": array of tool name strings (can be empty)

JSON array:"#,
        count = config.count,
        agent_name = &agent.name,
        domain = domain,
        keyword_list = keyword_list,
        tool_list = tool_list,
        prompt_excerpt = agent.system_prompt.chars().take(400).collect::<String>(),
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;
    let response = client
        .post(format!("{}/chat/completions", base_url))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "model": config.llm_model,
            "messages": [
                {"role": "system", "content": "You are a test scenario generator. Output only valid JSON with no markdown fences or extra text."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }))
        .send()
        .await
        .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

    let provider = if base_url.contains("nvidia") {
        "nvidia"
    } else {
        "openai"
    };
    if !response.status().is_success() {
        return Err(AgentForgeError::LlmError {
            provider: provider.to_string(),
            message: format!("HTTP {}", response.status()),
        });
    }

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("[]");

    parse_llm_scenarios(content, agent, config)
}

fn parse_llm_scenarios(
    content: &str,
    agent: &AgentFile,
    config: &DomainSeededConfig,
) -> Result<Vec<Scenario>> {
    // Strip markdown code fences that some models add despite instructions not to
    let json_str = strip_markdown_fences(content);

    let parsed: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| AgentForgeError::ParseError(format!("LLM returned invalid JSON: {e}")))?;

    // Handle both {"scenarios": [...]} and [...] formats
    let arr = if let Some(arr) = parsed.as_array() {
        arr
    } else if let Some(arr) = parsed.get("scenarios").and_then(|s| s.as_array()) {
        arr
    } else {
        return Err(AgentForgeError::ParseError(
            "LLM response did not contain a scenarios array".to_string(),
        ));
    };

    let scenarios = arr
        .iter()
        .take(config.count)
        .map(|item| {
            let user_message = item["user_message"]
                .as_str()
                .unwrap_or("Help me with a task.")
                .to_string();
            let pass_criteria = item["pass_criteria"]
                .as_str()
                .unwrap_or("Agent provides a relevant and accurate response.")
                .to_string();
            let difficulty = match item["difficulty"].as_str().unwrap_or("medium") {
                "easy" => DifficultyTier::Easy,
                "hard" => DifficultyTier::Hard,
                "edge" => DifficultyTier::Edge,
                _ => DifficultyTier::Medium,
            };

            let expected_tools: Vec<agentforge_core::ExpectedToolCall> = item["expected_tools"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str())
                        .filter(|name| agent.tools.iter().any(|tool| &tool.name == name))
                        .map(|name| agentforge_core::ExpectedToolCall {
                            tool_name: name.to_string(),
                            required: true,
                            argument_schema: agent
                                .tools
                                .iter()
                                .find(|t| t.name == name)
                                .map(|t| t.parameters.clone()),
                        })
                        .collect()
                })
                .unwrap_or_default();

            Scenario {
                id: Uuid::new_v4(),
                agent_id: config.agent_id,
                input: ScenarioInput {
                    user_message,
                    conversation_history: vec![],
                    context: None,
                },
                expected: ScenarioExpected {
                    tool_calls: expected_tools,
                    output_schema: agent.output_schema.clone(),
                    pass_criteria,
                    min_turns: Some(1),
                    max_turns: Some(5),
                },
                difficulty,
                domain: agent.eval_hints.as_ref().and_then(|h| h.domain.clone()),
                source: ScenarioSource::DomainSeeded,
                tags: vec!["domain_seeded".to_string()],
                created_at: Utc::now(),
            }
        })
        .collect();

    Ok(scenarios)
}

/// Strip markdown code fences that some models add around JSON output.
fn strip_markdown_fences(s: &str) -> &str {
    let s = s.trim();
    // Handle ```json ... ``` or ``` ... ```
    if let Some(inner) = s.strip_prefix("```json").or_else(|| s.strip_prefix("```")) {
        if let Some(end) = inner.rfind("```") {
            return inner[..end].trim();
        }
    }
    s
}

/// Heuristic domain scenario generation (no LLM required).
fn generate_heuristic(
    agent: &AgentFile,
    config: &DomainSeededConfig,
    domain: &str,
    keywords: &[String],
) -> Result<Vec<Scenario>> {
    let domain_scenarios = domain_templates(domain, keywords, &agent.tools);
    let mut scenarios = Vec::new();

    for (msg, criteria, difficulty) in domain_scenarios.into_iter().cycle().take(config.count) {
        // Associate scenarios with the most relevant tools based on keywords
        let relevant_tools: Vec<agentforge_core::ExpectedToolCall> = agent
            .tools
            .iter()
            .filter(|t| {
                let desc_lower = t.description.to_lowercase();
                keywords.iter().any(|kw| desc_lower.contains(kw.as_str()))
            })
            .take(2)
            .map(|t| agentforge_core::ExpectedToolCall {
                tool_name: t.name.clone(),
                required: false,
                argument_schema: Some(t.parameters.clone()),
            })
            .collect();

        scenarios.push(Scenario {
            id: Uuid::new_v4(),
            agent_id: config.agent_id,
            input: ScenarioInput {
                user_message: msg,
                conversation_history: vec![],
                context: None,
            },
            expected: ScenarioExpected {
                tool_calls: relevant_tools,
                output_schema: agent.output_schema.clone(),
                pass_criteria: criteria,
                min_turns: Some(1),
                max_turns: Some(5),
            },
            difficulty,
            domain: Some(domain.to_string()),
            source: ScenarioSource::DomainSeeded,
            tags: vec!["domain_seeded".to_string(), format!("domain:{}", domain)],
            created_at: Utc::now(),
        });
    }

    Ok(scenarios)
}

fn domain_templates(
    domain: &str,
    keywords: &[String],
    tools: &[agentforge_core::ToolDefinition],
) -> Vec<(String, String, DifficultyTier)> {
    let keyword_sample = keywords
        .first()
        .cloned()
        .unwrap_or_else(|| "the requested task".to_string());
    let _tool_name = tools
        .first()
        .map(|t| t.name.as_str())
        .unwrap_or("the appropriate tool");

    match domain {
        "customer_support" => vec![
            (
                "I placed an order yesterday and haven't received a confirmation email. What's going on?".to_string(),
                "Agent should look up the order status and provide a clear update with next steps.".to_string(),
                DifficultyTier::Easy,
            ),
            (
                "My order ORD-12345678 is showing as 'processing' for 5 days. Can you expedite it?".to_string(),
                "Agent should check the order status and either escalate or provide accurate timeline.".to_string(),
                DifficultyTier::Medium,
            ),
            (
                "I want to cancel my order but I already got the shipping confirmation. Is it too late?".to_string(),
                "Agent should check cancellation eligibility and explain the options accurately.".to_string(),
                DifficultyTier::Hard,
            ),
            (
                "I ordered the wrong size. Can I exchange it AND get a refund for the shipping?".to_string(),
                "Agent should handle the multi-part request, checking eligibility for each part.".to_string(),
                DifficultyTier::Hard,
            ),
        ],
        "code_assistant" => vec![
            (
                "Write a function to check if a string is a palindrome.".to_string(),
                "Agent should produce correct, readable code with a brief explanation.".to_string(),
                DifficultyTier::Easy,
            ),
            (
                "Explain what this code does and find any bugs: `for i in range(len(arr)): arr[i] = arr[i+1]`".to_string(),
                "Agent should identify the off-by-one error and explain the code correctly.".to_string(),
                DifficultyTier::Medium,
            ),
        ],
        _ => vec![
            (
                format!("I need help with {keyword_sample}."),
                "Agent should provide a relevant, accurate, and helpful response to the domain-specific request.".to_string(),
                DifficultyTier::Easy,
            ),
            (
                "Can you help me understand how this works?".to_string(),
                "Agent should provide a clear, accurate explanation relevant to its domain.".to_string(),
                DifficultyTier::Medium,
            ),
            (
                format!("I've tried everything and still can't solve this issue with {keyword_sample}."),
                "Agent should acknowledge the frustration, diagnose the issue, and provide actionable steps.".to_string(),
                DifficultyTier::Hard,
            ),
        ],
    }
}

/// Extract domain-relevant keywords from the system prompt.
pub fn extract_domain_keywords(system_prompt: &str) -> Vec<String> {
    // Remove common stop words and extract content words
    let stop_words = [
        "you", "are", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "is", "be", "will", "should", "always", "never", "must", "can", "have", "has",
        "do", "does", "your", "our", "their", "its", "this", "that", "these", "those", "not", "no",
    ];

    let word_re = Regex::new(r"\b[a-zA-Z]{4,}\b").expect("valid regex");
    let mut keywords: Vec<String> = word_re
        .find_iter(system_prompt)
        .map(|m| m.as_str().to_lowercase())
        .filter(|w| !stop_words.contains(&w.as_str()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    keywords.sort();
    keywords.truncate(20);
    keywords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_domain_keywords() {
        let prompt = "You are a helpful customer support agent for an e-commerce platform. \
            Always verify order details before making changes.";
        let keywords = extract_domain_keywords(prompt);
        assert!(keywords
            .iter()
            .any(|k| k.contains("customer") || k.contains("support") || k.contains("order")));
    }

    #[test]
    fn heuristic_generates_correct_count() {
        use agentforge_core::{ModelConfig, ModelProvider};
        let agent = AgentFile {
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
            system_prompt: "You are a customer support agent.".to_string(),
            tools: vec![],
            output_schema: None,
            constraints: vec![],
            eval_hints: Some(agentforge_core::EvalHints {
                domain: Some("customer_support".to_string()),
                ..agentforge_core::EvalHints::default()
            }),
            metadata: None,
        };
        let config = DomainSeededConfig {
            count: 5,
            agent_id: Uuid::new_v4(),
            llm_base_url: None,
            llm_api_key: None,
            llm_model: "gpt-4o-mini".to_string(),
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let scenarios = rt
            .block_on(generate_domain_seeded_scenarios(&agent, &config))
            .unwrap();
        assert_eq!(scenarios.len(), 5);
    }
}
