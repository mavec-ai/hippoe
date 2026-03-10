pub mod config;
pub mod embedding;
pub mod error;
pub mod hippocampus;
pub mod memory;
pub mod recall;
pub mod storage;
pub mod types;

pub use config::Config;
pub use embedding::{EmbeddingError, EmbeddingProvider, EmbeddingResult, SharedEmbeddingProvider};
pub use hippocampus::{Hippocampus, HippocampusBuilder};
pub use memory::{
    Association, AssociationBuilder, AssociationGraph, Memory, MemoryBuilder, MemoryContent,
    MemoryMetadata, TemporalContext, compute_association_strength,
};
pub use recall::scorer::{
    cosine_similarity, cosine_similarity_batch, similarity, similarity_batch,
};
pub use recall::strategy::{
    CognitiveRetrieval, RetrievalContext, RetrievalMatch, RetrievalScores, RetrievalStrategy,
};
pub use recall::{MemoryQuery, MemoryQueryBuilder};
pub use storage::{InMemoryStorage, Storage};
pub use types::{Embedding, Emotion, Id, Link, LinkKind, Timestamp};

#[cfg(feature = "sqlite")]
pub use storage::SqliteStorage;

#[cfg(feature = "embedding-fastembed")]
pub use embedding::FastEmbedProvider;

#[cfg(feature = "embedding-ollama")]
pub use embedding::OllamaProvider;

#[cfg(feature = "embedding-openai")]
pub use embedding::OpenAIProvider;
