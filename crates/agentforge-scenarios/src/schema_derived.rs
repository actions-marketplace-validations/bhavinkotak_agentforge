use agentforge_core::{
    AgentFile, DifficultyTier, ExpectedToolCall, Result, Scenario, ScenarioExpected, ScenarioInput,
    ScenarioSource, ToolDefinition,
};
use chrono::Utc;
use uuid::Uuid;

/// Generate scenarios that exercise every tool and output field.
pub fn generate_schema_derived_scenarios(
    agent: &AgentFile,
    count: usize,
    agent_id: Uuid,
) -> Result<Vec<Scenario>> {
    let mut scenarios = Vec::new();

    // 1. One happy-path scenario per tool
    for tool in &agent.tools {
        if scenarios.len() >= count {
            break;
        }
        scenarios.push(happy_path_for_tool(agent, tool, agent_id));
    }

    // 2. If we have output schema, generate scenarios for required fields
    if let Some(schema) = &agent.output_schema {
        if scenarios.len() < count {
            if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
                for field in required {
                    if scenarios.len() >= count {
                        break;
                    }
                    if let Some(field_name) = field.as_str() {
                        scenarios.push(output_field_scenario(agent, field_name, agent_id));
                    }
                }
            }
        }
    }

    // 3. Fill remaining with multi-tool scenarios
    let mut rng_seed = 42u64;
    while scenarios.len() < count && !agent.tools.is_empty() {
        scenarios.push(multi_tool_scenario(agent, agent_id, rng_seed));
        rng_seed += 1;
    }

    // If still not enough (agent has no tools), generate generic task scenarios
    while scenarios.len() < count {
        scenarios.push(generic_task_scenario(agent, agent_id, scenarios.len()));
    }

    Ok(scenarios)
}

fn happy_path_for_tool(agent: &AgentFile, tool: &ToolDefinition, agent_id: Uuid) -> Scenario {
    // Real users never say "use the X tool" — they describe their problem.
    // Generate a concrete, realistic prompt so the model has enough context to
    // produce meaningful tool arguments and demonstrate actual competence.
    let user_message = realistic_user_message(tool, agent);

    let pass_criteria =
        "The agent should understand the user's request, select the most appropriate tool(s), \
         call them with valid and complete arguments, and provide a clear, actionable response."
            .to_string();

    Scenario {
        id: Uuid::new_v4(),
        agent_id,
        input: ScenarioInput {
            user_message,
            conversation_history: vec![],
            context: None,
        },
        expected: ScenarioExpected {
            tool_calls: vec![ExpectedToolCall {
                tool_name: tool.name.clone(),
                required: true,
                argument_schema: Some(tool.parameters.clone()),
            }],
            output_schema: agent.output_schema.clone(),
            pass_criteria,
            min_turns: Some(1),
            max_turns: Some(5),
        },
        difficulty: DifficultyTier::Easy,
        domain: agent.eval_hints.as_ref().and_then(|h| h.domain.clone()),
        source: ScenarioSource::SchemaDerived,
        tags: vec!["happy_path".to_string(), format!("tool:{}", tool.name)],
        created_at: Utc::now(),
    }
}

/// Extract a short domain label from agent metadata or name.
fn domain_label(agent: &AgentFile) -> String {
    agent
        .metadata
        .as_ref()
        .and_then(|m| m.get("description"))
        .and_then(|v| v.as_str())
        .map(|d| {
            let end = d.find([',', '.', ';']).unwrap_or(d.len().min(80));
            d[..end].trim().to_string()
        })
        .filter(|s| s.len() > 5)
        .unwrap_or_else(|| agent.name.clone())
}

/// Generate a realistic user message for a tool without naming the tool explicitly.
fn realistic_user_message(tool: &ToolDefinition, agent: &AgentFile) -> String {
    // Copilot agents expose capability strings in x-copilot-capability.
    // Use that to understand *what* the tool does rather than its short name.
    let capability = tool
        .parameters
        .get("x-copilot-capability")
        .and_then(|v| v.as_str())
        .unwrap_or(tool.name.as_str());

    let cap = capability.to_lowercase();
    let agent_lower = agent.name.to_lowercase();
    let domain = domain_label(agent);

    let is_ci_agent = agent_lower.contains("action")
        || agent_lower.contains("workflow")
        || agent_lower.contains("ci")
        || agent_lower.contains("cd");

    if cap.starts_with("github") {
        if is_ci_agent {
            "Create a secure GitHub Actions CI workflow for a Node.js project. \
             It should trigger on pull requests, run the test suite, upload coverage \
             reports, and block merges on failure. Pin every action reference to a \
             full commit SHA and apply least-privilege permissions throughout."
                .to_string()
        } else {
            format!(
                "I need help with a GitHub repository task related to {domain}. \
                 Please help me configure or update the relevant GitHub settings."
            )
        }
    } else if cap.contains("search") && cap.contains("codebase") {
        if is_ci_agent {
            "Search the repository for all GitHub Actions workflow files under \
             .github/workflows/. List each file and flag any that use mutable action \
             references (like @main or @v3) instead of pinned commit SHAs."
                .to_string()
        } else {
            format!(
                "Search the codebase for all {domain} configuration files and flag \
                 anything that looks outdated or inconsistent with best practices."
            )
        }
    } else if cap.contains("edit") || cap.contains("write") {
        if is_ci_agent {
            "Update the workflow at .github/workflows/ci.yml to pin all action \
             references to their full commit SHAs, restrict permissions to the \
             minimum required, and add a concurrency group to cancel stale PR runs."
                .to_string()
        } else {
            format!(
                "Update the {domain} configuration files to follow current best \
                 practices. Please make the necessary edits."
            )
        }
    } else if cap.contains("execute") || cap.contains("run") || cap.contains("terminal") {
        if is_ci_agent {
            "Run actionlint on all workflow files in this repository to validate \
             their syntax and logic. Report any errors or warnings found."
                .to_string()
        } else {
            format!(
                "Run the standard validation and lint checks for {domain}. \
                 Report any issues found."
            )
        }
    } else if cap.contains("read") || cap.contains("file") {
        if is_ci_agent {
            "Read the file .github/workflows/ci.yml and give me a detailed review: \
             what does it do, are the permissions correct, are actions pinned, and \
             what improvements would you recommend?"
                .to_string()
        } else {
            format!(
                "Read the current {domain} configuration and summarize what it does, \
                 highlighting any potential issues or improvements."
            )
        }
    } else if cap.contains("web") || cap.contains("fetch") || cap.contains("browse") {
        format!(
            "Look up the latest official guidance on {domain} and give me a concise \
             summary of the key recommendations I should follow."
        )
    } else if cap.contains("context") || cap.contains("docs") {
        format!(
            "What are the current best practices for {domain}? Please look up the \
             relevant documentation and give me a concise, actionable summary."
        )
    } else if tool.description.len() > 20 && !tool.description.starts_with("Copilot capability") {
        // Structured tool with a real description — use it as a hint
        format!(
            "I need to accomplish a task involving: {}. Please help me complete it \
             as part of my work on {domain}.",
            tool.description.chars().take(60).collect::<String>()
        )
    } else {
        format!("I need assistance with a {domain} task. Please help me get started.")
    }
}

/// Composite task message for multi-tool scenarios.
fn composite_task_message(agent: &AgentFile) -> String {
    let agent_lower = agent.name.to_lowercase();
    let domain = domain_label(agent);

    if agent_lower.contains("action")
        || agent_lower.contains("workflow")
        || agent_lower.contains("ci")
    {
        "Set up a complete, secure CI/CD pipeline for my project. Search for any \
         existing workflow files, review them for security issues (mutable action \
         tags, overly broad permissions, hardcoded secrets), update them to follow \
         current security best practices, and finally run actionlint to validate \
         the result."
            .to_string()
    } else if agent_lower.contains("security") || agent_lower.contains("safe") {
        "Audit the security posture of this repository. Read the relevant \
         configuration files, identify any vulnerabilities or misconfigurations, \
         and apply the necessary fixes."
            .to_string()
    } else {
        format!(
            "Help me complete a multi-step {domain} project task: first discover \
             the relevant files, review their current state, then apply any needed \
             improvements and verify the result."
        )
    }
}

fn output_field_scenario(agent: &AgentFile, field_name: &str, agent_id: Uuid) -> Scenario {
    Scenario {
        id: Uuid::new_v4(),
        agent_id,
        input: ScenarioInput {
            user_message: format!(
                "Complete a task that requires the agent to produce a '{}' field in its output.",
                field_name
            ),
            conversation_history: vec![],
            context: None,
        },
        expected: ScenarioExpected {
            tool_calls: vec![],
            output_schema: agent.output_schema.clone(),
            pass_criteria: format!(
                "The agent's output must include a valid '{}' field matching the schema.",
                field_name
            ),
            min_turns: Some(1),
            max_turns: Some(3),
        },
        difficulty: DifficultyTier::Easy,
        domain: agent.eval_hints.as_ref().and_then(|h| h.domain.clone()),
        source: ScenarioSource::SchemaDerived,
        tags: vec!["output_field".to_string(), format!("field:{}", field_name)],
        created_at: Utc::now(),
    }
}

fn multi_tool_scenario(agent: &AgentFile, agent_id: Uuid, seed: u64) -> Scenario {
    // Pick two tools (or repeat if only one)
    let n = agent.tools.len();
    let tool_a = &agent.tools[(seed as usize) % n];
    let tool_b = &agent.tools[(seed as usize + 1) % n];

    let expected_calls = if tool_a.name == tool_b.name {
        vec![ExpectedToolCall {
            tool_name: tool_a.name.clone(),
            required: true,
            argument_schema: Some(tool_a.parameters.clone()),
        }]
    } else {
        vec![
            ExpectedToolCall {
                tool_name: tool_a.name.clone(),
                required: true,
                argument_schema: Some(tool_a.parameters.clone()),
            },
            ExpectedToolCall {
                tool_name: tool_b.name.clone(),
                required: false,
                argument_schema: Some(tool_b.parameters.clone()),
            },
        ]
    };

    Scenario {
        id: Uuid::new_v4(),
        agent_id,
        input: ScenarioInput {
            user_message: composite_task_message(agent),
            conversation_history: vec![],
            context: None,
        },
        expected: ScenarioExpected {
            tool_calls: expected_calls,
            output_schema: agent.output_schema.clone(),
            pass_criteria: "Agent should call the appropriate tools in a logical order and provide a complete response.".to_string(),
            min_turns: Some(1),
            max_turns: Some(6),
        },
        difficulty: DifficultyTier::Medium,
        domain: agent.eval_hints.as_ref().and_then(|h| h.domain.clone()),
        source: ScenarioSource::SchemaDerived,
        tags: vec!["multi_tool".to_string()],
        created_at: Utc::now(),
    }
}

fn generic_task_scenario(agent: &AgentFile, agent_id: Uuid, index: usize) -> Scenario {
    let agent_lower = agent.name.to_lowercase();
    let domain = domain_label(agent);

    let messages: &[&str] = if agent_lower.contains("action")
        || agent_lower.contains("workflow")
        || agent_lower.contains("ci")
    {
        &[
            "What security risks should I be aware of when setting up GitHub Actions \
             for a public repository?",
            "Walk me through migrating from mutable action version tags like @v3 to \
             pinned commit SHAs across all my workflows.",
            "Explain how OIDC works with GitHub Actions and when I should use it \
             instead of long-lived credentials.",
            "What is the recommended way to handle deployment secrets in GitHub \
             Actions workflows to avoid accidental exposure?",
            "How should I structure my workflow permissions to follow the principle \
             of least privilege?",
        ]
    } else {
        &[
            "What best practices should I follow for this type of configuration?",
            "Help me understand the key concepts I need to know to work effectively \
             with this tool.",
            "What are the most common mistakes people make and how can I avoid them?",
            "Give me a step-by-step guide for getting started with the basics.",
            "What should I check first when troubleshooting unexpected behavior?",
        ]
    };

    let msg = messages[index % messages.len()];

    Scenario {
        id: Uuid::new_v4(),
        agent_id,
        input: ScenarioInput {
            user_message: msg.to_string(),
            conversation_history: vec![],
            context: None,
        },
        expected: ScenarioExpected {
            tool_calls: vec![],
            output_schema: agent.output_schema.clone(),
            pass_criteria: format!(
                "The agent should provide a helpful, accurate, and actionable response \
                 that draws on its expertise in {domain}."
            ),
            min_turns: Some(1),
            max_turns: Some(5),
        },
        difficulty: DifficultyTier::Easy,
        domain: agent.eval_hints.as_ref().and_then(|h| h.domain.clone()),
        source: ScenarioSource::SchemaDerived,
        tags: vec!["generic".to_string()],
        created_at: Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentforge_core::{ModelConfig, ModelProvider};

    fn make_agent_with_tools() -> AgentFile {
        AgentFile {
            agentforge_schema_version: "1".to_string(),
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model: ModelConfig {
                provider: ModelProvider::Openai,
                model_id: "gpt-4o".to_string(),
                temperature: None,
                max_tokens: None,
                top_p: None,
            },
            system_prompt: "You are helpful.".to_string(),
            tools: vec![
                ToolDefinition {
                    name: "tool_a".to_string(),
                    description: "Tool A".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                },
                ToolDefinition {
                    name: "tool_b".to_string(),
                    description: "Tool B".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                },
            ],
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"]
            })),
            constraints: vec![],
            eval_hints: None,
            metadata: None,
        }
    }

    #[test]
    fn generates_at_least_one_per_tool() {
        let agent = make_agent_with_tools();
        let id = Uuid::new_v4();
        let scenarios = generate_schema_derived_scenarios(&agent, 5, id).unwrap();
        assert_eq!(scenarios.len(), 5);
        // At least one scenario should target tool_a
        assert!(scenarios
            .iter()
            .any(|s| s.tags.iter().any(|t| t.contains("tool_a"))));
    }

    #[test]
    fn generates_exact_count() {
        let agent = make_agent_with_tools();
        let id = Uuid::new_v4();
        for n in [1, 5, 10, 20] {
            let s = generate_schema_derived_scenarios(&agent, n, id).unwrap();
            assert_eq!(s.len(), n, "Expected {n} scenarios");
        }
    }

    #[test]
    fn all_scenarios_have_unique_ids() {
        let agent = make_agent_with_tools();
        let id = Uuid::new_v4();
        let scenarios = generate_schema_derived_scenarios(&agent, 10, id).unwrap();
        let ids: std::collections::HashSet<_> = scenarios.iter().map(|s| s.id).collect();
        assert_eq!(ids.len(), 10);
    }
}
