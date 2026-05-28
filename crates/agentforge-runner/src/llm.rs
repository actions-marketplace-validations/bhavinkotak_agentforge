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
        let model =
            std::env::var("AGENTFORGE_OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2:3b".to_string());
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

// ─── AWS SigV4 helpers ─────────────────────────────────────────────────────

/// Percent-encode a single URI path segment per RFC 3986.
/// Only unreserved characters (A-Z, a-z, 0-9, '-', '.', '_', '~') are left
/// as-is; everything else (including ':') is encoded as `%XX`.
fn percent_encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char);
            }
            b => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Compute SHA-256 and return the lowercase hex string.
fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(data))
}

/// Compute HMAC-SHA256 without the `hmac` crate (saves a dependency).
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let h = sha2::Sha256::digest(key);
        k[..h.len()].copy_from_slice(&h);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let ipad: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = k.iter().map(|b| b ^ 0x5c).collect();
    let inner_hash = {
        let mut buf = ipad;
        buf.extend_from_slice(data);
        sha2::Sha256::digest(&buf)
    };
    let outer = {
        let mut buf = opad;
        buf.extend_from_slice(&inner_hash);
        sha2::Sha256::digest(&buf)
    };
    outer.into()
}

/// Derive the SigV4 signing key:
/// `HMAC(HMAC(HMAC(HMAC("AWS4" + secret, date), region), service), "aws4_request")`
fn derive_signing_key(secret: &str, date: &str, region: &str, service: &str) -> [u8; 32] {
    let k = hmac_sha256(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k = hmac_sha256(&k, region.as_bytes());
    let k = hmac_sha256(&k, service.as_bytes());
    hmac_sha256(&k, b"aws4_request")
}

/// Build the `Authorization` header value and `x-amz-date` timestamp for an
/// AWS SigV4 signed request.  Returns `(authorization_header, x_amz_date)`.
#[allow(clippy::too_many_arguments)]
fn sigv4_authorization(
    access_key_id: &str,
    secret_access_key: &str,
    session_token: Option<&str>,
    region: &str,
    service: &str,
    host: &str,
    method: &str,
    uri_path: &str,
    payload: &[u8],
    now: chrono::DateTime<chrono::Utc>,
) -> (String, String) {
    let date_time = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = &date_time[..8];

    // Canonical headers must be sorted lexicographically.
    let mut headers: Vec<(String, String)> = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("host".to_string(), host.to_string()),
        ("x-amz-date".to_string(), date_time.clone()),
    ];
    if let Some(token) = session_token {
        headers.push(("x-amz-security-token".to_string(), token.to_string()));
    }
    headers.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_headers: String = headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed_headers: String = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_request = format!(
        "{method}\n{uri_path}\n\n{canonical_headers}\n{signed_headers}\n{}",
        sha256_hex(payload)
    );

    let credential_scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{date_time}\n{credential_scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let signing_key = derive_signing_key(secret_access_key, date, region, service);
    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    let auth = format!(
        "AWS4-HMAC-SHA256 Credential={access_key_id}/{credential_scope}, \
         SignedHeaders={signed_headers}, Signature={signature}"
    );
    (auth, date_time)
}

// ─── BedrockClient ─────────────────────────────────────────────────────────

/// AWS Bedrock Converse API client.
///
/// Uses the [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
/// for a uniform messages interface across all Bedrock-hosted model families
/// (Anthropic Claude, Meta Llama, Mistral, …). Authenticates via AWS SigV4.
///
/// # Credential resolution
///
/// Reads `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from the environment.
/// `AWS_SESSION_TOKEN` is included when present (required for temporary
/// credentials / STS AssumeRole). Instance roles, IRSA, and IAM Roles Anywhere
/// are supported — ensure the process has a role with `bedrock:InvokeModel`
/// permission; the env vars can remain unset if the SDK credential chain
/// (instance metadata) supplies them at runtime instead.
pub struct BedrockClient {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
    region: String,
    model: String,
    client: reqwest::Client,
}

impl BedrockClient {
    pub fn new(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
        region: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token,
            region: region.into(),
            model: model.into(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("valid reqwest client"),
        }
    }

    /// Construct from standard AWS environment variables.
    ///
    /// Returns `None` when `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` are
    /// not set.
    pub fn from_env() -> Option<Self> {
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID").ok()?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok()?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());
        let model = std::env::var("AGENTFORGE_BEDROCK_MODEL")
            .unwrap_or_else(|_| "anthropic.claude-3-haiku-20240307-v1:0".to_string());
        Some(Self::new(
            access_key_id,
            secret_access_key,
            session_token,
            region,
            model,
        ))
    }
}

#[async_trait]
impl LlmClient for BedrockClient {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse> {
        let body = build_converse_request(&request)?;
        let body_bytes =
            serde_json::to_vec(&body).map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

        let encoded_model = percent_encode_path_segment(&self.model);
        let uri_path = format!("/model/{encoded_model}/converse");
        let host = format!("bedrock-runtime.{}.amazonaws.com", self.region);
        let endpoint = format!("https://{host}{uri_path}");

        let now = chrono::Utc::now();
        let (auth_header, amz_date) = sigv4_authorization(
            &self.access_key_id,
            &self.secret_access_key,
            self.session_token.as_deref(),
            &self.region,
            "bedrock",
            &host,
            "POST",
            &uri_path,
            &body_bytes,
            now,
        );

        let start = std::time::Instant::now();
        let mut req_builder = self
            .client
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .header("x-amz-date", &amz_date)
            .header("Authorization", &auth_header)
            .body(body_bytes);

        if let Some(token) = &self.session_token {
            req_builder = req_builder.header("x-amz-security-token", token);
        }

        let resp = req_builder
            .send()
            .await
            .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

        let latency_ms = start.elapsed().as_millis() as u64;

        if resp.status() == 429 {
            return Err(AgentForgeError::RateLimitExceeded {
                provider: "bedrock".to_string(),
            });
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            if status.is_server_error() {
                return Err(AgentForgeError::HttpError(format!(
                    "bedrock: HTTP {status}: {text}"
                )));
            }
            return Err(AgentForgeError::LlmError {
                provider: "bedrock".to_string(),
                message: format!("HTTP {status}: {text}"),
            });
        }

        let raw: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AgentForgeError::HttpError(e.to_string()))?;

        parse_converse_response(raw, latency_ms, &self.model)
    }

    fn provider_name(&self) -> &str {
        "bedrock"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

/// Convert an [`LlmRequest`] into a Bedrock Converse API request body.
///
/// The Converse API format differs from OpenAI / Anthropic in several ways:
/// - System messages go into a top-level `system` array, not inside `messages`.
/// - Content is an array of typed blocks (`text`, `toolUse`, `toolResult`).
/// - Tool definitions use `toolSpec` / `inputSchema.json` instead of
///   `function.parameters`.
fn build_converse_request(request: &LlmRequest) -> Result<serde_json::Value> {
    // Extract system messages — Converse API keeps them separate.
    let system: Vec<serde_json::Value> = request
        .messages
        .iter()
        .filter(|m| m.role == LlmRole::System)
        .filter_map(|m| m.content.as_deref())
        .map(|text| serde_json::json!({"text": text}))
        .collect();

    // Convert non-system messages to Converse content blocks.
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for msg in request
        .messages
        .iter()
        .filter(|m| m.role != LlmRole::System)
    {
        let converse_msg = match msg.role {
            LlmRole::User => serde_json::json!({
                "role": "user",
                "content": [{"text": msg.content.as_deref().unwrap_or("")}]
            }),
            LlmRole::Assistant => {
                let mut content: Vec<serde_json::Value> = Vec::new();
                if let Some(text) = &msg.content {
                    if !text.is_empty() {
                        content.push(serde_json::json!({"text": text}));
                    }
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({}));
                        content.push(serde_json::json!({
                            "toolUse": {
                                "toolUseId": tc.id,
                                "name": tc.function.name,
                                "input": input
                            }
                        }));
                    }
                }
                serde_json::json!({"role": "assistant", "content": content})
            }
            // Tool results are delivered back as user-role messages.
            LlmRole::Tool => serde_json::json!({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": msg.tool_call_id.as_deref().unwrap_or(""),
                        "content": [{"text": msg.content.as_deref().unwrap_or("")}]
                    }
                }]
            }),
            LlmRole::System => unreachable!("filtered above"),
        };
        messages.push(converse_msg);
    }

    let mut body = serde_json::json!({"messages": messages});

    if !system.is_empty() {
        body["system"] = serde_json::json!(system);
    }

    // inferenceConfig — only include keys that were explicitly set.
    let mut ic = serde_json::Map::new();
    if let Some(mt) = request.max_tokens {
        ic.insert("maxTokens".to_string(), serde_json::json!(mt));
    }
    if let Some(temp) = request.temperature {
        ic.insert("temperature".to_string(), serde_json::json!(temp));
    }
    if let Some(top_p) = request.top_p {
        ic.insert("topP".to_string(), serde_json::json!(top_p));
    }
    if !ic.is_empty() {
        body["inferenceConfig"] = serde_json::Value::Object(ic);
    }

    // toolConfig — convert OpenAI function format → Bedrock toolSpec format.
    if let Some(tools) = &request.tools {
        let tool_specs: Vec<serde_json::Value> = tools
            .iter()
            .filter_map(|t| {
                let func = t.get("function")?;
                let name = func.get("name")?.as_str()?;
                let description = func
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let params = func
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type": "object"}));
                Some(serde_json::json!({
                    "toolSpec": {
                        "name": name,
                        "description": description,
                        "inputSchema": {"json": params}
                    }
                }))
            })
            .collect();
        if !tool_specs.is_empty() {
            body["toolConfig"] = serde_json::json!({"tools": tool_specs});
        }
    }

    Ok(body)
}

/// Parse a Bedrock Converse API response into an [`LlmResponse`].
fn parse_converse_response(
    raw: serde_json::Value,
    latency_ms: u64,
    model: &str,
) -> Result<LlmResponse> {
    let content_blocks = raw
        .get("output")
        .and_then(|o| o.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
        .ok_or_else(|| AgentForgeError::LlmError {
            provider: "bedrock".to_string(),
            message: "No output.message.content in Converse response".to_string(),
        })?;

    let mut text_content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in content_blocks {
        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
            text_content.push_str(text);
        } else if let Some(tu) = block.get("toolUse") {
            let id = tu
                .get("toolUseId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = tu
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = tu
                .get("input")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "{}".to_string());
            tool_calls.push(ToolCall {
                id,
                tool_type: "function".to_string(),
                function: ToolCallFunction { name, arguments },
            });
        }
    }

    let input_tokens = raw["usage"]["inputTokens"].as_u64().unwrap_or(0) as u32;
    let output_tokens = raw["usage"]["outputTokens"].as_u64().unwrap_or(0) as u32;
    let finish_reason = raw["stopReason"].as_str().unwrap_or("end_turn").to_string();

    Ok(LlmResponse {
        model: model.to_string(),
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

    // ── Bedrock helpers ──────────────────────────────────────────────────────

    #[test]
    fn percent_encode_colon_in_model_id() {
        let encoded = percent_encode_path_segment("anthropic.claude-3-haiku-20240307-v1:0");
        assert_eq!(encoded, "anthropic.claude-3-haiku-20240307-v1%3A0");
    }

    #[test]
    fn percent_encode_safe_chars_unchanged() {
        // All unreserved chars (RFC 3986) must pass through without encoding.
        let s = "meta.llama3-1-8b-instruct-v1~0_safe-chars.test";
        assert_eq!(percent_encode_path_segment(s), s);
    }

    #[test]
    fn hmac_sha256_produces_known_output() {
        // Known HMAC-SHA256 test vector (RFC 4231 Test Case 1).
        // Key = 0x0b * 20, Data = "Hi There"
        let key = vec![0x0bu8; 20];
        let data = b"Hi There";
        let result = hmac_sha256(&key, data);
        let hex = hex::encode(result);
        assert_eq!(
            hex,
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn sigv4_authorization_produces_well_formed_header() {
        use chrono::TimeZone;
        let now = chrono::Utc.with_ymd_and_hms(2026, 5, 25, 0, 0, 0).unwrap();
        let (auth, amz_date) = sigv4_authorization(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            "bedrock",
            "bedrock-runtime.us-east-1.amazonaws.com",
            "POST",
            "/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse",
            b"{}",
            now,
        );
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20260525/us-east-1/bedrock/aws4_request"),
            "auth header format wrong: {auth}"
        );
        assert!(auth.contains("SignedHeaders=content-type;host;x-amz-date"));
        assert!(auth.contains("Signature="));
        assert_eq!(amz_date, "20260525T000000Z");
    }

    #[test]
    fn sigv4_authorization_includes_security_token_header() {
        use chrono::TimeZone;
        let now = chrono::Utc.with_ymd_and_hms(2026, 5, 25, 0, 0, 0).unwrap();
        let (auth, _) = sigv4_authorization(
            "AKID",
            "SECRET",
            Some("SESSION_TOKEN"),
            "eu-west-1",
            "bedrock",
            "bedrock-runtime.eu-west-1.amazonaws.com",
            "POST",
            "/model/test/converse",
            b"{}",
            now,
        );
        assert!(
            auth.contains("x-amz-security-token"),
            "token must appear in SignedHeaders when present: {auth}"
        );
    }

    #[test]
    fn build_converse_request_extracts_system_prompt() {
        let req = LlmRequest {
            model: "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
            messages: vec![
                LlmMessage {
                    role: LlmRole::System,
                    content: Some("You are a helpful assistant.".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                LlmMessage {
                    role: LlmRole::User,
                    content: Some("Hello!".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            tools: None,
            temperature: Some(0.2),
            max_tokens: Some(512),
            top_p: None,
        };
        let body = build_converse_request(&req).unwrap();
        // System prompt → top-level "system" field
        let system = body["system"].as_array().unwrap();
        assert_eq!(
            system[0]["text"].as_str().unwrap(),
            "You are a helpful assistant."
        );
        // User message → messages[0]
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"].as_str().unwrap(), "user");
        assert_eq!(msgs[0]["content"][0]["text"].as_str().unwrap(), "Hello!");
        // inferenceConfig
        assert_eq!(body["inferenceConfig"]["maxTokens"].as_u64().unwrap(), 512);
        assert!((body["inferenceConfig"]["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn build_converse_request_converts_tools() {
        let req = LlmRequest {
            model: "test".to_string(),
            messages: vec![LlmMessage {
                role: LlmRole::User,
                content: Some("search for rust".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: Some(vec![json!({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            })]),
            temperature: None,
            max_tokens: None,
            top_p: None,
        };
        let body = build_converse_request(&req).unwrap();
        let tool_spec = &body["toolConfig"]["tools"][0]["toolSpec"];
        assert_eq!(tool_spec["name"].as_str().unwrap(), "web_search");
        assert_eq!(tool_spec["description"].as_str().unwrap(), "Search the web");
        assert!(tool_spec["inputSchema"]["json"].is_object());
    }

    #[test]
    fn parse_converse_response_text() {
        let raw = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Here is the answer."}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 20, "outputTokens": 10}
        });
        let resp =
            parse_converse_response(raw, 80, "anthropic.claude-3-haiku-20240307-v1:0").unwrap();
        assert_eq!(resp.message.content.as_deref(), Some("Here is the answer."));
        assert_eq!(resp.finish_reason, "end_turn");
        assert_eq!(resp.input_tokens, 20);
        assert_eq!(resp.output_tokens, 10);
        assert_eq!(resp.latency_ms, 80);
        assert_eq!(resp.model, "anthropic.claude-3-haiku-20240307-v1:0");
    }

    #[test]
    fn parse_converse_response_tool_use() {
        let raw = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{
                        "toolUse": {
                            "toolUseId": "tu_abc",
                            "name": "get_weather",
                            "input": {"location": "Seattle"}
                        }
                    }]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 30, "outputTokens": 15}
        });
        let resp = parse_converse_response(raw, 120, "test-model").unwrap();
        assert!(resp.message.content.is_none());
        let tcs = resp.message.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "tu_abc");
        assert_eq!(tcs[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&tcs[0].function.arguments).unwrap();
        assert_eq!(args["location"].as_str().unwrap(), "Seattle");
    }

    #[test]
    fn parse_converse_response_missing_content_is_error() {
        let raw = json!({"stopReason": "end_turn"});
        assert!(parse_converse_response(raw, 0, "test").is_err());
    }

    #[test]
    fn bedrock_client_from_env_returns_none_without_keys() {
        // With no AWS env vars set this should return None.
        std::env::remove_var("AWS_ACCESS_KEY_ID");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");
        assert!(BedrockClient::from_env().is_none());
    }
}
