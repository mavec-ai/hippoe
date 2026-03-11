//! Memory retrieval and scoring algorithms.
//!
//! This module provides the retrieval pipeline:
//!
//! - [`MemoryQuery`]: Query construction with filters and constraints
//! - [`CognitiveRetrieval`]: Hybrid retrieval strategy combining multiple factors
//! - [`RetrievalStrategy`]: Trait for custom retrieval implementations
//!
//! # Hybrid Retrieval Strategy
//!
//! CognitiveRetrieval combines multiple factors with configurable weights:
//!
//! | Factor | Default Weight | Description |
//! |--------|----------------|-------------|
//! | Similarity | 1.0 | MINERVA 2 cubed cosine similarity |
//! | Base-level | 0.8 | ACT-R activation based on access patterns |
//! | Spreading | 0.7 | Activation from associated memories |
//! | Emotional | 0.5 | Emotion-weighted memory boost |
//! | Contextual | 0.3 | Context match bonus |
//! | Temporal | 0.3 | TCM temporal context similarity |
//!
//! The final score combines factors multiplicatively with additive boosts:
//! - Core factors (similarity, base-level, spreading, emotional) combined via `combine_activations_multiplicative`
//! - Context and temporal factors applied as additive boosts: `base * (1 + context*weight) * (1 + temporal*weight)`
//!
//! # MINERVA 2 Similarity
//!
//! Similarity between probe and memory is computed as:
//! `similarity = max(0, cos³(probe, memory))`
//!
//! The cubic power emphasizes strong matches and suppresses weak ones.
//!
//! Reference: Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. DOI:10.1037/0033-295X.93.4.411

mod query;
pub mod scorer;
pub mod strategy;

pub use query::{MemoryQuery, MemoryQueryBuilder};
pub use scorer::{
    cosine_similarity, cosine_similarity_batch, retrieval_probability_batch, similarity,
    similarity_batch,
};
pub use strategy::{
    CognitiveRetrieval, RetrievalContext, RetrievalMatch, RetrievalScores, RetrievalStrategy,
    WorkingMemoryBoost,
};
