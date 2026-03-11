use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("empty probe embedding")]
    EmptyProbe,

    #[error("no memories provided")]
    NoMemories,

    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("invalid min_score: {0} (must be >= 0)")]
    InvalidMinScore(f64),

    #[error("invalid link strength: {0}")]
    InvalidLinkStrength(f64),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("invalid id format")]
    InvalidId,
}

pub type Result<T> = std::result::Result<T, Error>;
