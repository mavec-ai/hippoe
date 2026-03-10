use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

#[cfg(feature = "embedding-fastembed")]
mod fastembed;
#[cfg(feature = "embedding-fastembed")]
pub use fastembed::FastEmbedProvider;

#[cfg(feature = "embedding-ollama")]
mod ollama;
#[cfg(feature = "embedding-ollama")]
pub use ollama::OllamaProvider;

#[cfg(feature = "embedding-openai")]
mod openai;
#[cfg(feature = "embedding-openai")]
pub use openai::OpenAIProvider;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("embedding provider error: {0}")]
    Provider(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("invalid response: {0}")]
    InvalidResponse(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("configuration error: {0}")]
    Config(String),
}

pub type EmbeddingResult<T> = std::result::Result<T, EmbeddingError>;

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f64>>;

    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f64>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimensions(&self) -> usize;

    fn model_name(&self) -> &str;
}

pub type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider>;
