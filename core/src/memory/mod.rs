//! Memory core types and association management.
//!
//! This module provides the fundamental memory structures:
//!
//! - [`Memory`]: The core memory unit with content, embedding, and metadata
//! - [`MemoryBuilder`]: Builder pattern for constructing memories
//! - [`TemporalContext`]: Temporal Context Model (TCM) implementation
//! - [`ConsolidationState`]: Reconsolidation state machine
//! - [`AssociationGraph`]: Network of memory associations
//! - [`AssociationBuilder`]: Constructs associations between memories
//!
//! # Cognitive Models Implemented
//!
//! ## ACT-R Base-Level Activation
//! Memory activation decays with time but strengthens with access:
//! `A = ln(n+1) - d·ln(t)` where n is access count and t is time since creation.
//!
//! ## Temporal Context Model (TCM)
//! Maintains a drifting temporal context vector that updates with new memories,
//! enabling temporal associations between sequentially experienced items.
//!
//! ## Reconsolidation Theory
//! Memories can be in one of four states: Active, Consolidating, Stable, or Labile.
//! Retrieval can trigger labile state, allowing memory updates.
//!
//! References:
//! - Anderson, J. R. (1997). ACT-R. DOI: 10.1037/0003-066X.52.4.355
//! - Howard, M. W., & Kahana, M. J. (2002). TCM. DOI: 10.1006/jmps.2001.1388
//! - Nader, K. et al. (2000). Reconsolidation. DOI: 10.1038/35021052

mod graph;
mod links;
#[allow(clippy::module_inception)]
mod memory;

pub use graph::{AssociationEdge, AssociationGraph};
pub use links::{AssociationBuilder, compute_association_strength};
pub use memory::{
    Association, ConsolidationState, Memory, MemoryBuilder, MemoryContent, MemoryMetadata,
    TemporalContext, TemporalLink,
};
