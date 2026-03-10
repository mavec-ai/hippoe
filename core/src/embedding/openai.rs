use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{EmbeddingError, EmbeddingProvider, EmbeddingResult};

#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    api_key: String,
    model: String,
    dimensions: usize,
    base_url: String,
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self::new("text-embedding-3-small")
    }
}

impl OpenAIProvider {
    pub fn new(model: &str) -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let dimensions = Self::get_dimensions_for_model(model);
        Self {
            api_key,
            model: model.to_string(),
            dimensions,
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = api_key.to_string();
        self
    }

    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.trim_end_matches('/').to_string();
        self
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    fn get_dimensions_for_model(model: &str) -> usize {
        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        }
    }
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Input {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize)]
struct OpenAIResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
    index: usize,
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f64>> {
        if self.api_key.is_empty() {
            return Err(EmbeddingError::Config("OPENAI_API_KEY not set".to_string()));
        }

        let client = reqwest::Client::new();

        let request = OpenAIRequest {
            model: self.model.clone(),
            input: Input::Single(text.to_string()),
            dimensions: None,
        };

        let response = client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::Provider(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let openai_response: OpenAIResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::InvalidResponse(e.to_string()))?;

        openai_response
            .data
            .first()
            .map(|d| d.embedding.clone())
            .ok_or_else(|| EmbeddingError::InvalidResponse("no embedding returned".to_string()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f64>>> {
        if self.api_key.is_empty() {
            return Err(EmbeddingError::Config("OPENAI_API_KEY not set".to_string()));
        }

        let client = reqwest::Client::new();

        let request = OpenAIRequest {
            model: self.model.clone(),
            input: Input::Batch(texts.iter().map(|s| s.to_string()).collect()),
            dimensions: None,
        };

        let response = client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::Provider(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let openai_response: OpenAIResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::InvalidResponse(e.to_string()))?;

        let mut results = vec![Vec::new(); texts.len()];
        for item in openai_response.data {
            let idx = item.index;
            if idx < results.len() {
                results[idx] = item.embedding;
            }
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
        let provider = OpenAIProvider::default();
        assert_eq!(provider.model, "text-embedding-3-small");
        assert_eq!(provider.dimensions(), 1536);
    }

    #[test]
    fn test_large_model() {
        let provider = OpenAIProvider::new("text-embedding-3-large");
        assert_eq!(provider.dimensions(), 3072);
    }

    #[test]
    fn test_with_api_key() {
        let provider = OpenAIProvider::default().with_api_key("test-key");
        assert_eq!(provider.api_key, "test-key");
    }
}
