use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use super::{EmbeddingError, EmbeddingProvider, EmbeddingResult};

pub use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub struct FastEmbedProvider {
    model: Arc<Mutex<TextEmbedding>>,
    model_name: String,
    dimensions: usize,
}

impl FastEmbedProvider {
    pub fn new(model: EmbeddingModel) -> EmbeddingResult<Self> {
        let options = InitOptions::new(model.clone());
        let embedding_model =
            TextEmbedding::try_new(options).map_err(|e| EmbeddingError::Provider(e.to_string()))?;

        let dimensions = Self::get_dimensions_for_model(&model);
        let model_name = model.to_string();

        Ok(Self {
            model: Arc::new(Mutex::new(embedding_model)),
            model_name,
            dimensions,
        })
    }

    pub fn new_default() -> EmbeddingResult<Self> {
        Self::new(EmbeddingModel::BGESmallENV15)
    }

    pub fn with_model_name(model_name: &str, dimensions: usize) -> EmbeddingResult<Self> {
        let model = Self::model_from_name(model_name)?;
        let options = InitOptions::new(model.clone());
        let embedding_model =
            TextEmbedding::try_new(options).map_err(|e| EmbeddingError::Provider(e.to_string()))?;

        Ok(Self {
            model: Arc::new(Mutex::new(embedding_model)),
            model_name: model_name.to_string(),
            dimensions,
        })
    }

    fn model_from_name(name: &str) -> EmbeddingResult<EmbeddingModel> {
        match name {
            "BAAI/bge-small-en-v1.5" => Ok(EmbeddingModel::BGESmallENV15),
            "BAAI/bge-base-en-v1.5" => Ok(EmbeddingModel::BGEBaseENV15),
            "BAAI/bge-large-en-v1.5" => Ok(EmbeddingModel::BGELargeENV15),
            "sentence-transformers/all-MiniLM-L6-v2" => Ok(EmbeddingModel::AllMiniLML6V2),
            "sentence-transformers/all-MiniLM-L12-v2" => Ok(EmbeddingModel::AllMiniLML12V2),
            "nomic-ai/nomic-embed-text-v1.5" => Ok(EmbeddingModel::NomicEmbedTextV15),
            "mixedbread-ai/mxbai-embed-large-v1" => Ok(EmbeddingModel::MxbaiEmbedLargeV1),
            "intfloat/multilingual-e5-small" => Ok(EmbeddingModel::MultilingualE5Small),
            "intfloat/multilingual-e5-base" => Ok(EmbeddingModel::MultilingualE5Base),
            "intfloat/multilingual-e5-large" => Ok(EmbeddingModel::MultilingualE5Large),
            _ => Err(EmbeddingError::ModelNotFound(name.to_string())),
        }
    }

    fn get_dimensions_for_model(model: &EmbeddingModel) -> usize {
        match model {
            EmbeddingModel::BGESmallENV15
            | EmbeddingModel::BGESmallENV15Q
            | EmbeddingModel::AllMiniLML6V2
            | EmbeddingModel::AllMiniLML6V2Q => 384,

            EmbeddingModel::BGEBaseENV15
            | EmbeddingModel::BGEBaseENV15Q
            | EmbeddingModel::NomicEmbedTextV1
            | EmbeddingModel::NomicEmbedTextV15
            | EmbeddingModel::MultilingualE5Small
            | EmbeddingModel::MultilingualE5Large => 768,

            EmbeddingModel::BGELargeENV15
            | EmbeddingModel::BGELargeENV15Q
            | EmbeddingModel::AllMiniLML12V2
            | EmbeddingModel::MxbaiEmbedLargeV1
            | EmbeddingModel::MultilingualE5Base => 1024,

            _ => 768,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f64>> {
        let mut model = self.model.lock();
        let embeddings = model
            .embed(vec![text.to_string()], None)
            .map_err(|e| EmbeddingError::Provider(e.to_string()))?;

        embeddings
            .first()
            .map(|v| v.iter().map(|&f| f as f64).collect())
            .ok_or_else(|| EmbeddingError::InvalidResponse("no embedding returned".to_string()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f64>>> {
        let mut model = self.model.lock();
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let embeddings = model
            .embed(texts_owned, None)
            .map_err(|e| EmbeddingError::Provider(e.to_string()))?;

        Ok(embeddings
            .into_iter()
            .map(|v| v.into_iter().map(|f| f as f64).collect())
            .collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_dimensions() {
        assert_eq!(
            FastEmbedProvider::get_dimensions_for_model(&EmbeddingModel::BGESmallENV15),
            384
        );
        assert_eq!(
            FastEmbedProvider::get_dimensions_for_model(&EmbeddingModel::BGEBaseENV15),
            768
        );
        assert_eq!(
            FastEmbedProvider::get_dimensions_for_model(&EmbeddingModel::BGELargeENV15),
            1024
        );
    }

    #[test]
    fn test_model_from_name() {
        assert!(FastEmbedProvider::model_from_name("BAAI/bge-small-en-v1.5").is_ok());
        assert!(FastEmbedProvider::model_from_name("unknown-model").is_err());
    }
}
