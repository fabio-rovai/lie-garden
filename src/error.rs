use thiserror::Error;

#[derive(Error, Debug)]
pub enum GravRailError {
    #[error("lie group error: {0}")]
    LieGroup(String),
    #[error("gauge error: {0}")]
    Gauge(String),
    #[error("circuit error: {0}")]
    Circuit(String),
    #[error("crypto error: {0}")]
    Crypto(String),
    #[error("state error: {0}")]
    State(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("configuration error: {0}")]
    Config(String),
    #[error("confinement violation: {0}")]
    Confinement(String),
    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),
}

pub type Result<T> = std::result::Result<T, GravRailError>;
