pub mod hippocampus;
pub mod config;
pub mod embedding;
pub mod error;
pub mod memory;
pub mod recall;
pub mod storage;
pub mod types;

pub use hippocampus::{Hippocampus, HippocampusBuilder};
pub use config::Config;
pub use embedding::{EmbeddingError, EmbeddingProvider, EmbeddingResult, SharedEmbeddingProvider};
pub use memory::{
    Association, AssociationBuilder, AssociationGraph, Memory, MemoryBuilder, MemoryContent,
    MemoryMetadata, TemporalContext, compute_association_strength,
};
pub use recall::{MemoryQuery, MemoryQueryBuilder};
pub use recall::scorer::{similarity, similarity_batch};
pub use recall::strategy::{
    CognitiveRetrieval, RetrievalContext, RetrievalMatch, RetrievalScores, RetrievalStrategy,
};
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
