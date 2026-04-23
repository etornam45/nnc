use std::{fmt, path::PathBuf};

#[derive(Debug, Clone)]
pub enum CompileError {
    Io(String),
    Decode(String),
    InvalidModel(String),
    UnsupportedOp(String),
    MissingAttribute { node: String, attr: String },
    InvalidAttribute { node: String, attr: String, detail: String },
    InvalidShape(String),
    Validation(String),
    Codegen(String),
    Cli(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Io(msg) => write!(f, "io error: {msg}"),
            CompileError::Decode(msg) => write!(f, "decode error: {msg}"),
            CompileError::InvalidModel(msg) => write!(f, "invalid model: {msg}"),
            CompileError::UnsupportedOp(op) => write!(f, "unsupported op: {op}"),
            CompileError::MissingAttribute { node, attr } => {
                write!(f, "node '{node}' is missing required attribute '{attr}'")
            }
            CompileError::InvalidAttribute { node, attr, detail } => {
                write!(f, "invalid attribute '{attr}' on node '{node}': {detail}")
            }
            CompileError::InvalidShape(msg) => write!(f, "invalid shape: {msg}"),
            CompileError::Validation(msg) => write!(f, "validation failed: {msg}"),
            CompileError::Codegen(msg) => write!(f, "codegen failed: {msg}"),
            CompileError::Cli(msg) => write!(f, "cli error: {msg}"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(value: std::io::Error) -> Self {
        CompileError::Io(value.to_string())
    }
}

pub type CompileResult<T> = Result<T, CompileError>;

#[derive(Debug, Clone)]
pub struct CapabilityReport {
    pub model_path: PathBuf,
    pub supported_ops: Vec<String>,
    pub unsupported_ops: Vec<String>,
}
