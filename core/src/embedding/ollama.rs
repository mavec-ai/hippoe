use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{EmbeddingError, EmbeddingProvider, EmbeddingResult};

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    base_url: String,
    model: String,
    dimensions: usize,
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new("http://localhost:11434", "nomic-embed-text")
    }
}

impl OllamaProvider {
    pub fn new(base_url: &str, model: &str) -> Self {
        let dimensions = Self::get_dimensions_for_model(model);
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dimensions,
        }
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    fn get_dimensions_for_model(model: &str) -> usize {
        match model {
            m if m.starts_with("nomic-embed-text") => 768,
            m if m.starts_with("mxbai-embed-large") => 1024,
            m if m.starts_with("all-minilm") => 384,
            _ => 768,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct OllamaResponse {
    embedding: Vec<f64>,
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f64>> {
        let client = reqwest::Client::new();

        let request = OllamaRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let response = client
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::Provider(format!(
                "Ollama API error ({}): {}",
                status, body
            )));
        }

        let ollama_response: OllamaResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::InvalidResponse(e.to_string()))?;

        Ok(ollama_response.embedding)
    }

    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f64>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_provider() {
        let provider = OllamaProvider::default();
        assert_eq!(provider.base_url, "http://localhost:11434");
        assert_eq!(provider.model, "nomic-embed-text");
        assert_eq!(provider.dimensions(), 768);
    }

    #[test]
    fn test_custom_model() {
        let provider = OllamaProvider::new("http://localhost:11434", "mxbai-embed-large");
        assert_eq!(provider.dimensions(), 1024);
    }

    #[test]
    fn test_with_dimensions() {
        let provider = OllamaProvider::default().with_dimensions(512);
        assert_eq!(provider.dimensions(), 512);
    }
}
