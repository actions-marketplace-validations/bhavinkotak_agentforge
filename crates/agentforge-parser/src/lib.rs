pub mod detect;
pub mod formats;
pub mod parser;
pub mod validator;

pub use agentforge_core::AgentFileFormat;
pub use detect::detect_format;
pub use parser::{
    parse_agent_file, parse_agent_file_with_format, to_agent_version, ParsedAgentFile,
};
pub use validator::{validate_agent_file, ValidationResult};
