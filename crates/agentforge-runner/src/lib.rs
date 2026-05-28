pub mod llm;
pub mod runner;
pub mod worker;

pub use llm::{
    AnthropicClient, BedrockClient, LlmClient, LlmMessage, LlmRequest, LlmResponse, LlmRole,
    NvidiaClient, OllamaClient, OpenAiClient, ToolCall,
};
pub use runner::{AgentRunner, RunResult, RunnerConfig};

/// Type alias for a boxed, dynamically-dispatched LLM client.
pub type LlmClientBox = Box<dyn LlmClient>;
