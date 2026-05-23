use agentforge_core::{AgentForgeError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A message in the LLM conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: LlmRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LlmRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    /// Serialized as `"type"` to match the OpenAI / NVIDIA NIM wire format.
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String, // JSON string
}

/// An LLM request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
}

/// An LLM response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub model: String,
    pub message: LlmMessage,
    pub finish_reason: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    pub raw_response: serde_json::Value,
}

/// Trait for any LLM provider.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a completion request.
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse>;

    /// Provider name (for error messages and circular-bias detection).
    fn provider_name(&self) -> &str;

    /// The specific model ID this client is configured with.
    fn model_id(&self) -> &str;
}

/// OpenAI-compatible LLM client.
pub struct OpenAiClient {
    base_url: String,
    api_key: String,
    model: String,
    /// Provider name used in error messages. Defaults to "openai" but can be
    /// overridden when this client is used as the inner transport for another
    /// provider (e.g. NvidiaClient sets this to "nvidia").
    provider: String,
    client: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            provider: "openai".to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("valid reqwest client"),
        }
    }

    /// Create with a custom provider label (used when wrapping for another endpoint).
    pub fn new_with_provider(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            ..Self::new(base_url, api_key, model)
        }
    }

    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").ok()?;
        let model =
            std::env::var("AGENTFORGE_OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
        Some(Self::new("https://api.openai.com/v1", api_key, model))
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse> {
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                let mut obj = serde_json::json!({
                    "role": match m.role {
                        LlmRole::System => "system",
                        LlmRole::User => "user",
                        LlmRole::Assistant => "assistant",
                        LlmRole::Tool => "tool",
                    }
                });
                if let Some(content) = &m.content {
                    obj["content"] = serde_json::json!(content);
                } else if m.role == LlmRole::Assistant {
                    // Some vLLM backends (e.g. NVIDIA NIM Mistral) require an explicit
                    // `"content": null` on assistant tool-call messages; omitting the
                    // key entirely causes the template engine to misidentify the last
                    // message role and return HTTP 400.
                    obj["content"] = serde_json::Value::Null;
                }
                if let Some(tool_calls) = &m.tool_calls {
                    obj["tool_calls"] = serde_json::to_value(tool_calls).unwrap_or_default();
                }
                if let Some(tcid) = &m.tool_call_id {
                    obj["tool_call_id"] = serde_json::json!(tcid);
                }
                if let Some(name) = &m.name {
                    obj["name"] = serde_json::json!(name);
                }
                obj
            })
            .collect();

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
        });

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(mt) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(mt);
        }
        if let Some(tools) = &request.tools {
            body["tools"] = serde_json::json!(tools);
            // NVIDIA NIM only supports single tool calls per turn; disable parallel calls.
            if self.provider == "nvidia" {
                body["parallel_tool_calls"] = serde_json::json!(false);
            }
        }

        let start = std::time::Instant::now();
        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

        let latency_ms = start.elapsed().as_millis() as u64;

        if resp.status() == 429 {
            return Err(AgentForgeError::RateLimitExceeded {
                provider: self.provider.clone(),
            });
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            // 5xx errors are transient (gateway timeouts, overloaded servers);
            // wrap them as HttpError so the retry logic in the runner picks them up.
            if status.is_server_error() {
                return Err(AgentForgeError::HttpError(format!(
                    "{}: HTTP {status}: {text}",
                    self.provider
                )));
            }
            return Err(AgentForgeError::LlmError {
                provider: self.provider.clone(),
                message: format!("HTTP {status}: {text}"),
            });
        }

        let raw: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

        parse_openai_response(raw, latency_ms, &self.provider)
    }

    fn provider_name(&self) -> &str {
        &self.provider
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

fn parse_openai_response(
    raw: serde_json::Value,
    latency_ms: u64,
    provider: &str,
) -> Result<LlmResponse> {
    let choice = raw["choices"][0]
        .as_object()
        .ok_or_else(|| AgentForgeError::LlmError {
            provider: provider.to_string(),
            message: "No choices in response".to_string(),
        })?;

    let msg_val = &choice["message"];
    let role = match msg_val["role"].as_str().unwrap_or("assistant") {
        "user" => LlmRole::User,
        "system" => LlmRole::System,
        "tool" => LlmRole::Tool,
        _ => LlmRole::Assistant,
    };

    let content = msg_val["content"].as_str().map(String::from);

    // Debug-log the raw tool_calls array so we can diagnose format issues.
    if msg_val
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty())
    {
        tracing::debug!(
            raw_tool_calls = %msg_val["tool_calls"],
            "Raw tool_calls from API"
        );
    }

    let tool_calls: Option<Vec<ToolCall>> = msg_val
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|arr| {
            let parsed: Vec<ToolCall> = arr
                .iter()
                .filter_map(|tc| {
                    // Robustly parse `id`: string first, then integer (some vLLM
                    // backends like NVIDIA NIM return numeric IDs).
                    let id = tc["id"]
                        .as_str()
                        .map(str::to_owned)
                        .or_else(|| tc["id"].as_i64().map(|n| n.to_string()))
                        .or_else(|| tc["id"].as_u64().map(|n| n.to_string()))?;
                    let name = tc["function"]["name"].as_str()?.to_string();
                    // `arguments` may be a JSON string or an inlined object.
                    let arguments = match &tc["function"]["arguments"] {
                        serde_json::Value::String(s) => s.clone(),
                        v if !v.is_null() => v.to_string(),
                        _ => "{}".to_string(),
                    };
                    Some(ToolCall {
                        id,
                        tool_type: tc["type"].as_str().unwrap_or("function").to_string(),
                        function: ToolCallFunction { name, arguments },
                    })
                })
                .collect();
            if parsed.is_empty() && !arr.is_empty() {
                tracing::warn!(
                    raw_tool_calls = %serde_json::to_string(arr).unwrap_or_default(),
                    "Tool calls array was non-empty but all entries failed to parse — \
                     check the id/function.name field format"
                );
            }
            parsed
        })
        // Treat Some([]) the same as None so we don't leave an assistant
        // message as the final history entry without any tool results.
        .filter(|v| !v.is_empty());

    // Fallback: some models (e.g. Llama 4 Maverick) emit tool calls as a JSON
    // object in the `content` field instead of the structured `tool_calls`
    // array.  Detect this format and convert it so the agentic loop works.
    let tool_calls = if tool_calls.is_none() {
        content
            .as_deref()
            .and_then(try_parse_text_tool_call)
            .map(|tc| {
                tracing::debug!(
                    name = %tc[0].function.name,
                    "Parsed text-encoded tool call from content field (model doesn't use tool_calls API)"
                );
                tc
            })
    } else {
        tool_calls
    };

    let input_tokens = raw["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let output_tokens = raw["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
    let finish_reason = choice["finish_reason"]
        .as_str()
        .unwrap_or("stop")
        .to_string();
    let model = raw["model"].as_str().unwrap_or("unknown").to_string();

    Ok(LlmResponse {
        model,
        message: LlmMessage {
            role,
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        finish_reason,
        input_tokens,
        output_tokens,
        latency_ms,
        raw_response: raw,
    })
}

/// Try to parse a text-encoded tool call that some models (e.g. Llama 4
/// Maverick) emit in the `content` field instead of using the structured
/// `tool_calls` API.
///
/// Supported text format:
/// ```json
/// {"type": "function", "name": "toolName", "parameters": {...}}
/// ```
/// The `parameters` key is an alias for `arguments` used by certain models.
fn try_parse_text_tool_call(content: &str) -> Option<Vec<ToolCall>> {
    let v: serde_json::Value = serde_json::from_str(content.trim()).ok()?;
    // Must be an object with `"type": "function"` and a non-empty `name`.
    if v.get("type").and_then(|t| t.as_str()) != Some("function") {
        return None;
    }
    let name = v.get("name").and_then(|n| n.as_str())?.to_string();
    // Accept either `arguments` (OpenAI) or `parameters` (Llama 4).
    let arguments = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .map(|a| match a {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| "{}".to_string());
    Some(vec![ToolCall {
        id: uuid::Uuid::new_v4().to_string(),
        tool_type: "function".to_string(),
        function: ToolCallFunction { name, arguments },
    }])
}

/// Anthropic Claude client.
pub struct AnthropicClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("valid reqwest client"),
        }
    }

    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok()?;
        let model = std::env::var("AGENTFORGE_ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());
        Some(Self::new(api_key, model))
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse> {
        // Split system prompt from messages
        let system = request
            .messages
            .iter()
            .find(|m| m.role == LlmRole::System)
            .and_then(|m| m.content.clone());

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter(|m| m.role != LlmRole::System)
            .map(|m| {
                let role = match m.role {
                    LlmRole::User => "user",
                    LlmRole::Assistant => "assistant",
                    LlmRole::Tool => "user", // Anthropic uses user for tool results
                    LlmRole::System => "user",
                };
                serde_json::json!({
                    "role": role,
                    "content": m.content.as_deref().unwrap_or("")
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(2048),
        });

        if let Some(sys) = system {
            body["system"] = serde_json::json!(sys);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        // Convert OpenAI tool format to Anthropic format
        if let Some(tools) = &request.tools {
            let anthropic_tools: Vec<serde_json::Value> = tools.iter().filter_map(|t| {
                let func = t.get("function")?;
                Some(serde_json::json!({
                    "name": func.get("name")?,
                    "description": func.get("description").unwrap_or(&serde_json::json!("")),
                    "input_schema": func.get("parameters").unwrap_or(&serde_json::json!({"type": "object"}))
                }))
            }).collect();
            body["tools"] = serde_json::json!(anthropic_tools);
        }

        let start = std::time::Instant::now();
        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

        let latency_ms = start.elapsed().as_millis() as u64;

        if resp.status() == 429 {
            return Err(AgentForgeError::RateLimitExceeded {
                provider: "anthropic".to_string(),
            });
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AgentForgeError::LlmError {
                provider: "anthropic".to_string(),
                message: format!("HTTP {status}: {text}"),
            });
        }

        let raw: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

        parse_anthropic_response(raw, latency_ms)
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

fn parse_anthropic_response(raw: serde_json::Value, latency_ms: u64) -> Result<LlmResponse> {
    let content_blocks = raw["content"]
        .as_array()
        .ok_or_else(|| AgentForgeError::LlmError {
            provider: "anthropic".to_string(),
            message: "No content in response".to_string(),
        })?;

    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for block in content_blocks {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(t) = block["text"].as_str() {
                    text_content.push_str(t);
                }
            }
            Some("tool_use") => {
                tool_calls.push(ToolCall {
                    id: block["id"].as_str().unwrap_or("").to_string(),
                    tool_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: block["name"].as_str().unwrap_or("").to_string(),
                        arguments: serde_json::to_string(&block["input"])
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                });
            }
            _ => {}
        }
    }

    let input_tokens = raw["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
    let output_tokens = raw["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
    let finish_reason = raw["stop_reason"]
        .as_str()
        .unwrap_or("end_turn")
        .to_string();
    let model = raw["model"].as_str().unwrap_or("unknown").to_string();

    Ok(LlmResponse {
        model,
        message: LlmMessage {
            role: LlmRole::Assistant,
            content: if text_content.is_empty() {
                None
            } else {
                Some(text_content)
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            name: None,
        },
        finish_reason,
        input_tokens,
        output_tokens,
        latency_ms,
        raw_response: raw,
    })
}

/// NVIDIA NIM client.
///
/// NVIDIA NIM exposes a fully OpenAI-compatible `/chat/completions` endpoint at
/// `https://integrate.api.nvidia.com/v1`, so this client is a thin wrapper around
/// [`OpenAiClient`] with a different base URL and provider name.
///
/// # Free models available on build.nvidia.com
///
/// The following models are available under the free tier and require no credits:
///
/// | Model ID | Notes |
/// |---|---|
/// | `meta/llama-3.1-8b-instruct` | Default — fast, good for evals |
/// | `meta/llama-3.1-70b-instruct` | Higher quality, slower |
/// | `mistralai/mistral-7b-instruct-v0.3` | Compact, general purpose |
/// | `nvidia/nemotron-mini-4b-instruct` | NVIDIA-tuned, very fast |
/// | `microsoft/phi-3-mini-4k-instruct` | Efficient small model |
///
/// Set `AGENTFORGE_NVIDIA_MODEL` to override the default model.
pub struct NvidiaClient {
    inner: OpenAiClient,
}

impl NvidiaClient {
    /// Create a new NVIDIA NIM client targeting `build.nvidia.com`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: OpenAiClient::new_with_provider(
                "https://integrate.api.nvidia.com/v1",
                api_key,
                model,
                "nvidia",
            ),
        }
    }

    /// Construct from environment variables.
    ///
    /// Reads `NVIDIA_API_KEY` (required) and `AGENTFORGE_NVIDIA_MODEL`
    /// (optional, defaults to `meta/llama-3.1-8b-instruct`).
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("NVIDIA_API_KEY").ok()?;
        let model = std::env::var("AGENTFORGE_NVIDIA_MODEL")
            .unwrap_or_else(|_| "mistralai/mistral-small-4-119b-2603".to_string());
        Some(Self::new(api_key, model))
    }
}

#[async_trait]
impl LlmClient for NvidiaClient {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse> {
        // The agent fixture may specify an OpenAI model (e.g. "gpt-4o").
        // Always override with the configured NVIDIA NIM model so the request
        // is valid for the NVIDIA endpoint.
        let request = LlmRequest {
            model: self.inner.model_id().to_string(),
            ..request
        };
        self.inner.complete(request).await
    }

    fn provider_name(&self) -> &str {
        "nvidia"
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
}

/// Ollama local LLM client.
///
/// Talks to the Ollama OpenAI-compatible endpoint at `http://localhost:11434/v1`
/// (or a custom `AGENTFORGE_OLLAMA_BASE_URL`).  Requires no API key.  Always
/// overrides the model in the request with the configured Ollama model so that
/// agent files using cloud model IDs (e.g. `gpt-4o`) work transparently.
pub struct OllamaClient {
    inner: OpenAiClient,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: OpenAiClient::new_with_provider(base_url, "ollama", model, "ollama"),
        }
    }

    pub fn from_env() -> Self {
        let base_url = std::env::var("AGENTFORGE_OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
        let model = std::env::var("AGENTFORGE_OLLAMA_MODEL")
            .unwrap_or_else(|_| "llama3.2:3b".to_string());
        Self::new(base_url, model)
    }
}

#[async_trait]
impl LlmClient for OllamaClient {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse> {
        // Override the model with the configured Ollama model so that agent
        // files referencing cloud models (e.g. "gpt-4o") work transparently.
        let request = LlmRequest {
            model: self.inner.model_id().to_string(),
            ..request
        };
        self.inner.complete(request).await
    }

    fn provider_name(&self) -> &str {
        "ollama"
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_openai_response() {
        let raw = json!({
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        });
        let resp = parse_openai_response(raw, 100, "openai").unwrap();
        assert_eq!(resp.message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.input_tokens, 10);
        assert_eq!(resp.latency_ms, 100);
    }

    #[test]
    fn parses_openai_tool_call() {
        let raw = json!({
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": "{\"order_id\": \"ORD-123\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15}
        });
        let resp = parse_openai_response(raw, 200, "openai").unwrap();
        let tool_calls = resp.message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_order");
    }

    #[test]
    fn parses_anthropic_response() {
        let raw = json!({
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "text", "text": "I'll help you with that."}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 8}
        });
        let resp = parse_anthropic_response(raw, 150).unwrap();
        assert_eq!(
            resp.message.content.as_deref(),
            Some("I'll help you with that.")
        );
    }

    #[test]
    fn parses_anthropic_tool_use() {
        let raw = json!({
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "get_order",
                    "input": {"order_id": "ORD-123"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 12}
        });
        let resp = parse_anthropic_response(raw, 200).unwrap();
        let tool_calls = resp.message.tool_calls.unwrap();
        assert_eq!(tool_calls[0].function.name, "get_order");
    }
}
