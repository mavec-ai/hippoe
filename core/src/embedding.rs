use crate::error::Result;
use crate::types::Embedding;
use async_trait::async_trait;

/// Trait for providing embeddings from external models (e.g., OpenAI, Ollama, FastEmbed).
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Convert text into an embedding vector.
    async fn embed(&self, text: &str) -> Result<Embedding>;
}

/// A simple mock provider that generates deterministic embeddings.
/// Useful for testing and development without relying on external APIs.
pub struct MockProvider {
    pub dimensions: usize,
}

impl MockProvider {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

#[async_trait]
impl EmbeddingProvider for MockProvider {
    async fn embed(&self, text: &str) -> Result<Embedding> {
        let mut embedding = vec![0.0; self.dimensions];

        // Simple deterministic mapping based on text
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            let index = i % self.dimensions;
            embedding[index] += (b as f64) / 255.0;
        }

        // Normalize the vector (Cosine similarity requires normalized vectors)
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider() {
        let provider = MockProvider::new(3);

        let emb1 = provider.embed("hello").await.unwrap();
        let emb2 = provider.embed("world").await.unwrap();

        assert_eq!(emb1.len(), 3);
        assert_eq!(emb2.len(), 3);
        assert_ne!(emb1, emb2); // Should produce different embeddings
    }
}