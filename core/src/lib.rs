//! # hippoe-core
//!
//! A cognitive-inspired memory system implementing multiple theories from
//! computational cognitive science.
//!
//! ## Overview
//!
//! hippoe-core provides a memory architecture inspired by human memory research,
//! implementing:
//!
//! - **ACT-R Base-Level Activation** (Anderson, 1996): Memory strength based on
//!   access frequency and recency
//! - **MINERVA 2 Similarity** (Hintzman, 1986): Cubed cosine similarity for
//!   probe-to-trace matching
//! - **Temporal Context Model** (Howard & Kahana, 2002): Temporal associations
//!   and context drift
//! - **Reconsolidation Theory** (Nader et al., 2000): Memory updates upon retrieval
//! - **Spreading Activation** (Anderson, 1983): Activation propagation through
//!   association networks
//! - **Circumplex Model of Affect** (Russell, 1980): Valence/arousal emotion dimensions

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
