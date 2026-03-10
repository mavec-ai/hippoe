use std::collections::HashMap;

use async_trait::async_trait;

use crate::memory::{AssociationGraph, Memory, TemporalContext};
use crate::types::{Embedding, Id, LinkKind, Timestamp};

#[derive(Debug, Clone)]
pub struct RetrievalContext {
    pub current_time: Timestamp,
    pub query_embedding: Embedding,
    pub query_text: Option<String>,
    pub context_tags: Vec<String>,
    pub max_results: usize,
    pub min_threshold: f64,
    pub temporal_context: Option<TemporalContext>,
}

impl RetrievalContext {
    pub fn new(query_embedding: Embedding, current_time: Timestamp) -> Self {
        Self {
            current_time,
            query_embedding,
            query_text: None,
            context_tags: Vec::new(),
            max_results: 100,
            min_threshold: 0.0,
            temporal_context: None,
        }
    }

    pub fn with_query_text(mut self, text: impl Into<String>) -> Self {
        self.query_text = Some(text.into());
        self
    }

    pub fn with_context_tags(mut self, tags: Vec<String>) -> Self {
        self.context_tags = tags;
        self
    }

    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    pub fn with_min_threshold(mut self, threshold: f64) -> Self {
        self.min_threshold = threshold;
        self
    }

    pub fn with_temporal_context(mut self, ctx: TemporalContext) -> Self {
        self.temporal_context = Some(ctx);
        self
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalMatch {
    pub memory_id: Id,
    pub scores: RetrievalScores,
    pub probability: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RetrievalScores {
    pub similarity: f64,
    pub base_level: f64,
    pub spreading: f64,
    pub emotional: f64,
    pub contextual: f64,
    pub temporal: f64,
    pub total: f64,
}

#[async_trait]
pub trait RetrievalStrategy: Send + Sync {
    async fn retrieve(
        &self,
        memories: &[Memory],
        graph: &AssociationGraph,
        context: &RetrievalContext,
    ) -> Vec<RetrievalMatch>;
}

pub struct CognitiveRetrieval {
    similarity_weight: f64,
    base_level_weight: f64,
    spreading_weight: f64,
    emotional_weight: f64,
    contextual_weight: f64,
    temporal_weight: f64,
    spreading_depth: usize,
    spreading_decay: f64,
    semantic_link_weight: f64,
    episodic_link_weight: f64,
    temporal_link_weight: f64,
}

impl CognitiveRetrieval {
    pub fn new() -> Self {
        Self {
            similarity_weight: 1.0,
            base_level_weight: 0.8,
            spreading_weight: 0.7,
            emotional_weight: 0.5,
            contextual_weight: 0.3,
            temporal_weight: 0.3,
            spreading_depth: 3,
            spreading_decay: 0.5,
            semantic_link_weight: 1.0,
            episodic_link_weight: 0.7,
            temporal_link_weight: 0.4,
        }
    }

    pub fn with_weights(
        mut self,
        similarity: f64,
        base_level: f64,
        spreading: f64,
        emotional: f64,
        contextual: f64,
        temporal: f64,
    ) -> Self {
        self.similarity_weight = similarity;
        self.base_level_weight = base_level;
        self.spreading_weight = spreading;
        self.emotional_weight = emotional;
        self.contextual_weight = contextual;
        self.temporal_weight = temporal;
        self
    }

    pub fn with_spreading(mut self, depth: usize, decay: f64) -> Self {
        self.spreading_depth = depth;
        self.spreading_decay = decay;
        self
    }

    pub fn with_link_weights(mut self, semantic: f64, episodic: f64, temporal: f64) -> Self {
        self.semantic_link_weight = semantic;
        self.episodic_link_weight = episodic;
        self.temporal_link_weight = temporal;
        self
    }

    pub fn with_emotional_weight(mut self, weight: f64) -> Self {
        self.emotional_weight = weight;
        self
    }

    pub fn with_contextual_weight(mut self, weight: f64) -> Self {
        self.contextual_weight = weight;
        self
    }

    fn compute_similarity(&self, query: &[f64], memory: &[f64]) -> f64 {
        if query.len() != memory.len() {
            return 0.0;
        }

        let dot_product: f64 = query.iter().zip(memory.iter()).map(|(a, b)| a * b).sum();
        let query_norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt();
        let memory_norm: f64 = memory.iter().map(|x| x * x).sum::<f64>().sqrt();

        if query_norm > 0.0 && memory_norm > 0.0 {
            (dot_product / (query_norm * memory_norm)).max(0.0)
        } else {
            0.0
        }
    }

    fn compute_similarity_activation(&self, similarity: f64, emotional_weight: f64) -> f64 {
        let k = 5.0 + 3.0 * emotional_weight;
        1.0 / (1.0 + (-k * (similarity - 0.5)).exp())
    }

    fn compute_base_level(&self, memory: &Memory, current_time: Timestamp) -> f64 {
        memory.metadata.base_level_activation(current_time)
    }

    fn compute_spreading(
        &self,
        memory: &Memory,
        graph: &AssociationGraph,
        all_memories: &[Memory],
        context: &RetrievalContext,
    ) -> f64 {
        let mut total_spreading = 0.0;

        let id_to_idx: HashMap<Id, usize> = all_memories
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id, i))
            .collect();

        let incoming_edges = graph.get_edges_to(memory.id);
        
        for edge in incoming_edges {
            if let Some(&idx) = id_to_idx.get(&edge.from) {
                let source = &all_memories[idx];
                let source_outgoing_count = graph.get_edges_from(source.id).len().max(1) as f64;
                let fan_normalization = source_outgoing_count.sqrt();
                
                let link_weight = match edge.kind {
                    LinkKind::Semantic => self.semantic_link_weight,
                    LinkKind::Episodic => self.episodic_link_weight,
                    LinkKind::Temporal => self.temporal_link_weight,
                    LinkKind::Causal => 0.8,
                };
                
                let similarity = self.compute_similarity(&context.query_embedding, &source.embedding);
                let activation = self.compute_similarity_activation(similarity, source.metadata.emotional_weight.weight());
                total_spreading += activation * edge.strength * link_weight * self.spreading_decay / fan_normalization;
            }
        }

        total_spreading
    }

    fn compute_emotional(&self, memory: &Memory) -> f64 {
        memory.metadata.emotional_modulation()
    }

    fn compute_contextual(&self, memory: &Memory, context: &RetrievalContext) -> f64 {
        let mut score = 0.0;

        if let Some(ref query_text) = context.query_text
            && let Some(ref memory_text) = memory.content.text
        {
            let query_lower = query_text.to_lowercase();
            let memory_lower = memory_text.to_lowercase();

            if memory_lower.contains(&query_lower) {
                score += 0.5;
            }

            let query_words: Vec<&str> = query_lower.split_whitespace().collect();
            let memory_words: Vec<&str> = memory_lower.split_whitespace().collect();

            let overlap = query_words
                .iter()
                .filter(|w| memory_words.contains(w))
                .count();
            let total = query_words.len().max(1);

            score += (overlap as f64 / total as f64) * 0.5;
        }

        if !context.context_tags.is_empty() {
            let tag_overlap = context
                .context_tags
                .iter()
                .filter(|tag| memory.metadata.tags.contains(tag))
                .count();
            let total = context.context_tags.len().max(1);
            score += tag_overlap as f64 / total as f64;
        }

        score
    }

    fn compute_temporal(&self, memory: &Memory, current_time: Timestamp, context: &RetrievalContext) -> f64 {
        if let Some(ref temporal_ctx) = context.temporal_context {
            let tcm_similarity = temporal_ctx.similarity_to_embedding(&memory.embedding);
            
            let time_since_created = current_time.saturating_sub(memory.metadata.created_at) as f64;
            let recency_factor = (-memory.metadata.decay_rate * time_since_created / 1000.0).exp();
            
            tcm_similarity * 0.6 + recency_factor * 0.4
        } else {
            let time_since_access = current_time.saturating_sub(memory.metadata.last_accessed_at) as f64;
            (-memory.metadata.decay_rate * time_since_access / 1000.0).exp()
        }
    }
}

impl Default for CognitiveRetrieval {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RetrievalStrategy for CognitiveRetrieval {
    async fn retrieve(
        &self,
        memories: &[Memory],
        graph: &AssociationGraph,
        context: &RetrievalContext,
    ) -> Vec<RetrievalMatch> {
        let mut matches: Vec<RetrievalMatch> = memories
            .iter()
            .map(|memory| {
                let raw_similarity = self.compute_similarity(&context.query_embedding, &memory.embedding);
                let similarity = self.compute_similarity_activation(raw_similarity, memory.metadata.emotional_weight.weight());
                let base_level = self.compute_base_level(memory, context.current_time);
                let spreading = self.compute_spreading(memory, graph, memories, context);
                let emotional = self.compute_emotional(memory);
                let contextual = self.compute_contextual(memory, context);
                let temporal = self.compute_temporal(memory, context.current_time, context);

                let total = (similarity * self.similarity_weight)
                    + (base_level * self.base_level_weight)
                    + (spreading * self.spreading_weight)
                    + (emotional * self.emotional_weight)
                    + (contextual * self.contextual_weight)
                    + (temporal * self.temporal_weight);

                RetrievalMatch {
                    memory_id: memory.id,
                    scores: RetrievalScores {
                        similarity,
                        base_level,
                        spreading,
                        emotional,
                        contextual,
                        temporal,
                        total,
                    },
                    probability: 0.0,
                }
            })
            .collect();

        matches.sort_by(|a, b| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if matches.len() > context.max_results {
            matches.truncate(context.max_results);
        }

        matches.retain(|m| m.score() >= context.min_threshold);

        let exp_scores: Vec<f64> = matches.iter().map(|m| m.score().exp()).collect();
        let prob_total: f64 = exp_scores.iter().sum();

        for (m, exp_score) in matches.iter_mut().zip(exp_scores.iter()) {
            m.probability = exp_score / prob_total;
        }

        matches
    }
}

impl RetrievalMatch {
    pub fn score(&self) -> f64 {
        self.scores.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryBuilder, AssociationEdge};
    use crate::types::now;

    fn make_embedding(values: &[f64]) -> Embedding {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[tokio::test]
    async fn test_cognitive_retrieval() {
        let strategy = CognitiveRetrieval::new();
        let current_time = now();

        let mem1 = Memory::text(
            "test memory one",
            make_embedding(&[1.0, 0.0, 0.0]),
            current_time,
        );
        let mem2 = Memory::text(
            "test memory two",
            make_embedding(&[0.0, 1.0, 0.0]),
            current_time,
        );

        let memories = vec![mem1, mem2];
        let graph = AssociationGraph::new();

        let context = RetrievalContext::new(make_embedding(&[0.9, 0.1, 0.0]), current_time)
            .with_max_results(10);

        let results = strategy.retrieve(&memories, &graph, &context).await;

        assert_eq!(results.len(), 2);
        assert!(results[0].scores.similarity > results[1].scores.similarity);
    }

    #[tokio::test]
    async fn test_contextual_matching() {
        let strategy = CognitiveRetrieval::new();
        let current_time = now();

        let mem1 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("important meeting notes")
            .tag("work")
            .build();

        let mem2 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("weekend plans")
            .tag("personal")
            .build();

        let memories = vec![mem1, mem2];
        let graph = AssociationGraph::new();

        let context = RetrievalContext::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .with_query_text("meeting")
            .with_context_tags(vec!["work".to_string()]);

        let results = strategy.retrieve(&memories, &graph, &context).await;

        assert!(!results.is_empty());
        assert!(results[0].scores.contextual > 0.0);
    }

    #[test]
    fn test_base_level_activation_monotonic_decay() {
        let now_time = now();
        let mut metadata = crate::memory::MemoryMetadata::new(now_time);
        metadata.accessed(now_time);

        let activation_t1 = metadata.base_level_activation(now_time + 1000);
        let activation_t2 = metadata.base_level_activation(now_time + 2000);
        let activation_t3 = metadata.base_level_activation(now_time + 5000);
        let activation_t4 = metadata.base_level_activation(now_time + 10000);

        assert!(activation_t1 > 0.0, "activation at t1 should be positive");
        assert!(activation_t2 > 0.0, "activation at t2 should be positive");
        assert!(activation_t3 > 0.0, "activation at t3 should be positive");
        assert!(activation_t4 > 0.0, "activation at t4 should be positive");

        assert!(
            activation_t1 >= activation_t2,
            "activation should decay monotonically: {} >= {}",
            activation_t1,
            activation_t2
        );
        assert!(
            activation_t2 >= activation_t3,
            "activation should decay monotonically: {} >= {}",
            activation_t2,
            activation_t3
        );
        assert!(
            activation_t3 >= activation_t4,
            "activation should decay monotonically: {} >= {}",
            activation_t3,
            activation_t4
        );
    }

    #[test]
    fn test_base_level_activation_access_boost() {
        let now_time = now();
        
        let mut metadata1 = crate::memory::MemoryMetadata::new(now_time);
        metadata1.accessed(now_time);
        
        let mut metadata2 = crate::memory::MemoryMetadata::new(now_time);
        for _ in 0..10 {
            metadata2.accessed(now_time);
        }

        let activation1 = metadata1.base_level_activation(now_time + 1000);
        let activation2 = metadata2.base_level_activation(now_time + 1000);

        assert!(
            activation2 > activation1,
            "more accesses should yield higher activation: {} > {}",
            activation2,
            activation1
        );
    }

    #[tokio::test]
    async fn test_fan_effect_normalization() {
        let strategy = CognitiveRetrieval::new();
        let current_time = now();

        let central_mem = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("central concept")
            .build();

        let peripheral1 = MemoryBuilder::new(make_embedding(&[0.0, 1.0, 0.0]), current_time)
            .text("peripheral 1")
            .build();

        let peripheral2 = MemoryBuilder::new(make_embedding(&[0.0, 0.0, 1.0]), current_time)
            .text("peripheral 2")
            .build();

        let memories = vec![central_mem.clone(), peripheral1.clone(), peripheral2.clone()];
        let mut graph = AssociationGraph::new();

        graph.add_edge(AssociationEdge::new(central_mem.id, peripheral1.id, 0.8, crate::types::LinkKind::Semantic, current_time));
        graph.add_edge(AssociationEdge::new(central_mem.id, peripheral2.id, 0.8, crate::types::LinkKind::Semantic, current_time));
        graph.add_edge(AssociationEdge::new(peripheral1.id, peripheral2.id, 0.5, crate::types::LinkKind::Semantic, current_time));

        let context = RetrievalContext::new(make_embedding(&[1.0, 0.0, 0.0]), current_time);

        let results = strategy.retrieve(&memories, &graph, &context).await;

        let central_result = results.iter().find(|r| r.memory_id == central_mem.id);
        assert!(
            central_result.is_some(),
            "central memory should be retrieved"
        );

        if let Some(central) = central_result {
            assert!(
                central.scores.spreading >= 0.0,
                "spreading activation should be non-negative"
            );
            assert!(
                central.scores.spreading <= 1.0,
                "spreading activation should be bounded by 1.0"
            );
        }
    }

    #[tokio::test]
    async fn test_similarity_sigmoid_bounded() {
        let strategy = CognitiveRetrieval::new();
        let current_time = now();

        let mem_high_sim = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("high similarity")
            .build();

        let mem_low_sim = MemoryBuilder::new(make_embedding(&[0.0, 1.0, 0.0]), current_time)
            .text("low similarity")
            .build();

        let memories = vec![mem_high_sim.clone(), mem_low_sim.clone()];
        let graph = AssociationGraph::new();

        let context = RetrievalContext::new(make_embedding(&[1.0, 0.0, 0.0]), current_time);

        let results = strategy.retrieve(&memories, &graph, &context).await;

        for result in &results {
            assert!(
                result.scores.similarity >= 0.0 && result.scores.similarity <= 1.0,
                "similarity score should be bounded [0, 1]: got {}",
                result.scores.similarity
            );
        }

        let high_sim_result = results.iter().find(|r| r.memory_id == mem_high_sim.id);
        let low_sim_result = results.iter().find(|r| r.memory_id == mem_low_sim.id);

        if let (Some(high), Some(low)) = (high_sim_result, low_sim_result) {
            assert!(
                high.scores.similarity > low.scores.similarity,
                "high similarity memory should score higher: {} > {}",
                high.scores.similarity,
                low.scores.similarity
            );
        }
    }
}
