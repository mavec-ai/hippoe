//! Association building and management.
//!
//! This module provides [`AssociationBuilder`] for constructing associations
//! between memories based on multiple criteria:
//!
//! - **Semantic**: Content similarity via embedding cosine similarity
//! - **Episodic**: Shared context, tags, or keyword overlap
//! - **Temporal**: Temporal proximity within configurable time windows
//!
//! # Association Types
//!
//! | Kind | Threshold | Basis |
//! |------|-----------|-------|
//! | Semantic | 0.7 (default) | Embedding similarity |
//! | Episodic | 0.5 (default) | Context/tags/keywords |
//! | Temporal | 0.3 (default) | Time proximity (60s window) |
//!
//! The builder pattern allows fine-grained control over which association
//! types are enabled and their respective thresholds.

use crate::memory::{Association, AssociationGraph, Memory};
use crate::recall::similarity_batch;
use crate::types::{Id, LinkKind, Timestamp};

pub struct AssociationBuilder {
    semantic_threshold: f64,
    episodic_threshold: f64,
    temporal_threshold: f64,
    max_associations: usize,
    link_semantic: bool,
    link_episodic: bool,
    link_temporal: bool,
}

impl AssociationBuilder {
    pub fn new() -> Self {
        Self {
            semantic_threshold: 0.7,
            episodic_threshold: 0.5,
            temporal_threshold: 0.3,
            max_associations: 10,
            link_semantic: true,
            link_episodic: true,
            link_temporal: true,
        }
    }

    pub fn with_semantic_threshold(mut self, threshold: f64) -> Self {
        self.semantic_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_episodic_threshold(mut self, threshold: f64) -> Self {
        self.episodic_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_temporal_threshold(mut self, threshold: f64) -> Self {
        self.temporal_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_max_associations(mut self, max: usize) -> Self {
        self.max_associations = max;
        self
    }

    pub fn enable_semantic(mut self, enabled: bool) -> Self {
        self.link_semantic = enabled;
        self
    }

    pub fn enable_episodic(mut self, enabled: bool) -> Self {
        self.link_episodic = enabled;
        self
    }

    pub fn enable_temporal(mut self, enabled: bool) -> Self {
        self.link_temporal = enabled;
        self
    }

    pub fn build_associations(
        &self,
        new_memory: &Memory,
        existing_memories: &[Memory],
        graph: &mut AssociationGraph,
    ) {
        if existing_memories.is_empty() {
            return;
        }

        let mut associations = Vec::new();

        if self.link_semantic {
            let semantic = self.build_semantic_associations(new_memory, existing_memories);
            associations.extend(semantic);
        }

        if self.link_episodic {
            let episodic = self.build_episodic_associations(new_memory, existing_memories);
            associations.extend(episodic);
        }

        if self.link_temporal {
            let temporal = self.build_temporal_associations(new_memory, existing_memories);
            associations.extend(temporal);
        }

        associations.sort_by(|a, b| {
            b.strength
                .total_cmp(&a.strength)
                .then_with(|| a.target_id.0.cmp(&b.target_id.0))
        });
        associations.truncate(self.max_associations);

        for association in associations {
            let edge = crate::memory::graph::AssociationEdge::new(
                new_memory.id,
                association.target_id,
                association.strength,
                association.kind,
                association.created_at,
            );
            graph.add_edge(edge);
        }
    }

    fn build_semantic_associations(
        &self,
        new_memory: &Memory,
        existing_memories: &[Memory],
    ) -> Vec<Association> {
        let embeddings: Vec<&[f64]> = existing_memories
            .iter()
            .map(|m| m.embedding.as_slice())
            .collect();

        let similarities = similarity_batch(&new_memory.embedding, &embeddings);

        let current_time = new_memory.metadata.created_at;

        let mut result = Vec::new();
        for (i, sim) in similarities.iter().enumerate() {
            if *sim >= self.semantic_threshold {
                result.push(Association::semantic(
                    existing_memories[i].id,
                    *sim,
                    current_time,
                ));
            }
        }
        result
    }

    fn build_episodic_associations(
        &self,
        new_memory: &Memory,
        existing_memories: &[Memory],
    ) -> Vec<Association> {
        let current_time = new_memory.metadata.created_at;
        let mut associations = Vec::new();

        for memory in existing_memories {
            if memory.metadata.context == new_memory.metadata.context
                && memory.metadata.context.is_some()
            {
                let strength = self.episodic_threshold.max(0.6);
                associations.push(Association::episodic(memory.id, strength, current_time));
            }

            let tag_overlap = new_memory
                .metadata
                .tags
                .iter()
                .filter(|tag| memory.metadata.tags.contains(tag))
                .count();

            if tag_overlap > 0 {
                let total_tags = new_memory
                    .metadata
                    .tags
                    .len()
                    .max(memory.metadata.tags.len());
                let strength =
                    (tag_overlap as f64 / total_tags as f64).max(self.episodic_threshold);
                associations.push(Association::episodic(memory.id, strength, current_time));
            }

            if let (Some(new_text), Some(existing_text)) =
                (&new_memory.content.text, &memory.content.text)
            {
                let keyword_overlap = self.compute_keyword_overlap(new_text, existing_text);
                if keyword_overlap > self.episodic_threshold {
                    associations.push(Association::episodic(
                        memory.id,
                        keyword_overlap,
                        current_time,
                    ));
                }
            }
        }

        associations
    }

    fn build_temporal_associations(
        &self,
        new_memory: &Memory,
        existing_memories: &[Memory],
    ) -> Vec<Association> {
        let current_time = new_memory.metadata.created_at;
        let time_window = 60_000;

        let mut temporal_memories: Vec<(f64, &Memory)> = existing_memories
            .iter()
            .filter_map(|memory| {
                let time_diff = current_time.saturating_sub(memory.metadata.created_at);
                if time_diff <= time_window {
                    let recency = 1.0 - (time_diff as f64 / time_window as f64);
                    Some((recency, memory))
                } else {
                    None
                }
            })
            .collect();

        temporal_memories.sort_by(|a, b| b.0.total_cmp(&a.0));
        temporal_memories.truncate(5);

        temporal_memories
            .into_iter()
            .filter(|(recency, _)| *recency >= self.temporal_threshold)
            .map(|(recency, memory)| {
                Association::temporal(
                    memory.id,
                    recency.max(self.temporal_threshold),
                    current_time,
                )
            })
            .collect()
    }

    fn compute_keyword_overlap(&self, text1: &str, text2: &str) -> f64 {
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();
        let words1: std::collections::HashSet<&str> = text1_lower.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2_lower.split_whitespace().collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    pub fn build_bidirectional(
        &self,
        memory1: &Memory,
        memory2: &Memory,
        strength: f64,
        kind: LinkKind,
        graph: &mut AssociationGraph,
    ) {
        let edge1 = crate::memory::graph::AssociationEdge::new(
            memory1.id,
            memory2.id,
            strength,
            kind,
            memory1.metadata.created_at,
        );
        let edge2 = crate::memory::graph::AssociationEdge::new(
            memory2.id,
            memory1.id,
            strength,
            kind,
            memory2.metadata.created_at,
        );

        graph.add_edge(edge1);
        graph.add_edge(edge2);
    }

    pub fn strengthen_association(
        &self,
        graph: &mut AssociationGraph,
        from: Id,
        to: Id,
        kind: LinkKind,
        increment: f64,
        at: Timestamp,
    ) {
        if let Some(edge) = graph.get_edge(from, to, kind) {
            let new_strength = (edge.strength + increment).min(1.0);
            let new_edge =
                crate::memory::graph::AssociationEdge::new(from, to, new_strength, kind, at);
            graph.add_edge(new_edge);
        }
    }
}

impl Default for AssociationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn compute_association_strength(memory1: &Memory, memory2: &Memory) -> f64 {
    let embedding_similarity = compute_cosine_similarity(&memory1.embedding, &memory2.embedding);

    let context_similarity = match (&memory1.metadata.context, &memory2.metadata.context) {
        (Some(c1), Some(c2)) if c1 == c2 => 0.3,
        _ => 0.0,
    };

    let tag_similarity = {
        let overlap = memory1
            .metadata
            .tags
            .iter()
            .filter(|tag| memory2.metadata.tags.contains(tag))
            .count();
        let total = memory1
            .metadata
            .tags
            .len()
            .max(memory2.metadata.tags.len())
            .max(1);
        (overlap as f64 / total as f64) * 0.2
    };

    (embedding_similarity + context_similarity + tag_similarity).min(1.0)
}

fn compute_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        (dot_product / (norm_a * norm_b)).max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Embedding;
    use crate::memory::MemoryBuilder;
    use crate::types::now;

    fn make_embedding(values: &[f64]) -> Embedding {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[test]
    fn test_semantic_associations() {
        let builder = AssociationBuilder::new()
            .with_semantic_threshold(0.7)
            .enable_semantic(true)
            .enable_episodic(false)
            .enable_temporal(false);

        let current_time = now();

        let mem1 = Memory::text("memory 1", make_embedding(&[1.0, 0.0, 0.0]), current_time);
        let mem2 = Memory::text("memory 2", make_embedding(&[0.95, 0.1, 0.0]), current_time);
        let mem3 = Memory::text("memory 3", make_embedding(&[0.1, 0.9, 0.0]), current_time);

        let existing = vec![mem1.clone(), mem3.clone()];
        let mut graph = AssociationGraph::new();

        builder.build_associations(&mem2, &existing, &mut graph);

        let edges_from_mem2 = graph.get_edges_from(mem2.id);
        assert!(!edges_from_mem2.is_empty());

        let has_semantic_to_mem1 = edges_from_mem2
            .iter()
            .any(|e| e.to == mem1.id && e.kind == LinkKind::Semantic);
        assert!(has_semantic_to_mem1);
    }

    #[test]
    fn test_episodic_associations() {
        let builder = AssociationBuilder::new()
            .enable_semantic(false)
            .enable_episodic(true)
            .enable_temporal(false);

        let current_time = now();

        let mem1 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("work meeting")
            .context("work")
            .build();

        let mem2 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("project discussion")
            .context("work")
            .build();

        let mem3 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("weekend plans")
            .context("personal")
            .build();

        let existing = vec![mem1.clone(), mem3.clone()];
        let mut graph = AssociationGraph::new();

        builder.build_associations(&mem2, &existing, &mut graph);

        let edges = graph.get_edges_from(mem2.id);
        let has_episodic_to_mem1 = edges
            .iter()
            .any(|e| e.to == mem1.id && e.kind == LinkKind::Episodic);
        assert!(has_episodic_to_mem1);
    }

    #[test]
    fn test_temporal_associations() {
        let builder = AssociationBuilder::new()
            .enable_semantic(false)
            .enable_episodic(false)
            .enable_temporal(true);

        let base_time = now();

        let mem1 = Memory::text("memory 1", make_embedding(&[1.0, 0.0, 0.0]), base_time);

        let mem2 = {
            let mut m = Memory::text(
                "memory 2",
                make_embedding(&[1.0, 0.0, 0.0]),
                base_time + 30_000,
            );
            m.metadata.created_at = base_time + 30_000;
            m
        };

        let mem3 = {
            let mut m = Memory::text(
                "memory 3",
                make_embedding(&[1.0, 0.0, 0.0]),
                base_time + 120_000,
            );
            m.metadata.created_at = base_time + 120_000;
            m
        };

        let existing = vec![mem1.clone()];
        let mut graph = AssociationGraph::new();

        builder.build_associations(&mem2, &existing, &mut graph);

        let edges = graph.get_edges_from(mem2.id);
        let has_temporal = edges.iter().any(|e| e.kind == LinkKind::Temporal);
        assert!(has_temporal);

        let mut graph2 = AssociationGraph::new();
        builder.build_associations(&mem3, &existing, &mut graph2);
        let edges2 = graph2.get_edges_from(mem3.id);
        assert!(edges2.is_empty());
    }

    #[test]
    fn test_bidirectional_association() {
        let builder = AssociationBuilder::new();
        let current_time = now();

        let mem1 = Memory::text("memory 1", make_embedding(&[1.0, 0.0, 0.0]), current_time);
        let mem2 = Memory::text("memory 2", make_embedding(&[0.9, 0.1, 0.0]), current_time);

        let mut graph = AssociationGraph::new();

        builder.build_bidirectional(&mem1, &mem2, 0.8, LinkKind::Semantic, &mut graph);

        assert!(
            graph
                .get_edge(mem1.id, mem2.id, LinkKind::Semantic)
                .is_some()
        );
        assert!(
            graph
                .get_edge(mem2.id, mem1.id, LinkKind::Semantic)
                .is_some()
        );
    }

    #[test]
    fn test_max_associations_limit() {
        let builder = AssociationBuilder::new()
            .with_max_associations(2)
            .with_semantic_threshold(0.0)
            .enable_semantic(true)
            .enable_episodic(false)
            .enable_temporal(false);

        let current_time = now();
        let mem1 = Memory::text("m1", make_embedding(&[1.0, 0.0, 0.0]), current_time);
        let mem2 = Memory::text("m2", make_embedding(&[0.9, 0.1, 0.0]), current_time);
        let mem3 = Memory::text("m3", make_embedding(&[0.8, 0.2, 0.0]), current_time);
        let mem4 = Memory::text("m4", make_embedding(&[0.7, 0.3, 0.0]), current_time);

        let existing = vec![mem1, mem2, mem3];
        let mut graph = AssociationGraph::new();

        builder.build_associations(&mem4, &existing, &mut graph);

        let edges = graph.get_edges_from(mem4.id);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_compute_association_strength() {
        let current_time = now();

        let mem1 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("test memory")
            .context("work")
            .tag("important")
            .build();

        let mem2 = MemoryBuilder::new(make_embedding(&[0.95, 0.1, 0.0]), current_time)
            .text("another test")
            .context("work")
            .tag("important")
            .build();

        let strength = compute_association_strength(&mem1, &mem2);
        assert!(strength > 0.8);
    }
}
