use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use crate::memory::{AssociationGraph, Memory, TemporalContext};
use crate::recall::scorer::combine_activations_multiplicative;
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
    pub session_id: Option<String>,
    pub working_memory_accesses: HashMap<Id, usize>,
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
            session_id: None,
            working_memory_accesses: HashMap::new(),
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

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_working_memory_accesses(mut self, accesses: HashMap<Id, usize>) -> Self {
        self.working_memory_accesses = accesses;
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
    pub raw_similarity: f64,
    pub base_level: f64,
    pub spreading: f64,
    pub emotional: f64,
    pub contextual: f64,
    pub temporal: f64,
    pub working_memory_boost: f64,
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

#[derive(Debug, Clone)]
pub struct WorkingMemoryConfig {
    pub boost_factor: f64,
    pub decay_rate: f64,
    pub max_boost: f64,
    pub session_ttl_ms: u64,
    pub max_sessions: usize,
}

impl Default for WorkingMemoryConfig {
    fn default() -> Self {
        Self {
            boost_factor: 1.5,
            decay_rate: 0.15,
            max_boost: 2.0,
            session_ttl_ms: 3_600_000,
            max_sessions: 100,
        }
    }
}

#[derive(Debug, Clone)]
struct SessionMetadata {
    last_access: std::time::Instant,
    accesses: HashMap<Id, usize>,
}

#[derive(Debug, Default)]
pub struct WorkingMemoryBoost {
    sessions: Arc<RwLock<HashMap<String, SessionMetadata>>>,
    config: WorkingMemoryConfig,
}

impl WorkingMemoryBoost {
    pub fn new(config: WorkingMemoryConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    pub fn record_access(&self, memory_id: Id, session_id: &str) {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .entry(session_id.to_string())
            .or_insert(SessionMetadata {
                last_access: std::time::Instant::now(),
                accesses: HashMap::new(),
            });
        session.last_access = std::time::Instant::now();
        *session.accesses.entry(memory_id).or_insert(0) += 1;

        if sessions.len() > self.config.max_sessions {
            self.evict_oldest_session(&mut sessions);
        }
    }

    pub fn compute_boost(&self, memory_id: Id, session_id: Option<&str>) -> f64 {
        let sessions = self.sessions.read().unwrap();

        let Some(session_id) = session_id else {
            return 0.0;
        };

        let Some(session) = sessions.get(session_id) else {
            return 0.0;
        };

        let Some(&count) = session.accesses.get(&memory_id) else {
            return 0.0;
        };

        let raw_boost = (count as f64).ln().max(0.0) * 0.8;
        (1.0 + raw_boost).min(self.config.max_boost)
    }

    pub fn get_session_accesses(&self, session_id: &str) -> HashMap<Id, usize> {
        let sessions = self.sessions.read().unwrap();
        sessions
            .get(session_id)
            .map(|s| s.accesses.clone())
            .unwrap_or_default()
    }

    pub fn get_access_count(&self, memory_id: Id, session_id: &str) -> usize {
        let sessions = self.sessions.read().unwrap();
        sessions
            .get(session_id)
            .and_then(|s| s.accesses.get(&memory_id).copied())
            .unwrap_or(0)
    }

    pub fn decay_session(&self, session_id: &str) {
        let mut sessions = self.sessions.write().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            for count in session.accesses.values_mut() {
                *count = (*count as f64 * (1.0 - self.config.decay_rate)) as usize;
            }
            session.accesses.retain(|_, c| *c > 0);
        }
    }

    pub fn clear_session(&self, session_id: &str) {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(session_id);
    }

    pub fn cleanup_expired_sessions(&self) {
        let mut sessions = self.sessions.write().unwrap();
        let ttl = std::time::Duration::from_millis(self.config.session_ttl_ms);
        sessions.retain(|_, session| session.last_access.elapsed() < ttl);
    }

    fn evict_oldest_session(&self, sessions: &mut HashMap<String, SessionMetadata>) {
        if let Some((oldest_id, _)) = sessions.iter().min_by_key(|(_, s)| s.last_access) {
            let oldest_id = oldest_id.clone();
            sessions.remove(&oldest_id);
        }
    }
}

impl Clone for WorkingMemoryBoost {
    fn clone(&self) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(self.sessions.read().unwrap().clone())),
            config: self.config.clone(),
        }
    }
}

pub struct CognitiveRetrieval {
    similarity_weight: f64,
    base_level_weight: f64,
    spreading_weight: f64,
    emotional_weight: f64,
    contextual_weight: f64,
    temporal_weight: f64,
    working_memory_weight: f64,
    spreading_depth: usize,
    spreading_decay: f64,
    semantic_link_weight: f64,
    episodic_link_weight: f64,
    temporal_link_weight: f64,
    bidirectional_spreading: bool,
    working_memory: WorkingMemoryBoost,
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
            working_memory_weight: 0.6,
            spreading_depth: 3,
            spreading_decay: 0.5,
            semantic_link_weight: 1.0,
            episodic_link_weight: 0.7,
            temporal_link_weight: 0.4,
            bidirectional_spreading: false,
            working_memory: WorkingMemoryBoost::default(),
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

    pub fn with_working_memory(mut self, weight: f64, config: WorkingMemoryConfig) -> Self {
        self.working_memory_weight = weight;
        self.working_memory = WorkingMemoryBoost::new(config);
        self
    }

    pub fn with_bidirectional_spreading(mut self, enabled: bool) -> Self {
        self.bidirectional_spreading = enabled;
        self
    }

    pub fn working_memory(&self) -> &WorkingMemoryBoost {
        &self.working_memory
    }

    pub fn working_memory_mut(&mut self) -> &mut WorkingMemoryBoost {
        &mut self.working_memory
    }

    fn compute_wm_boost(
        &self,
        memory_id: Id,
        accesses: &HashMap<Id, usize>,
        session_id: Option<&str>,
    ) -> f64 {
        let Some(session_id) = session_id else {
            return 1.0;
        };

        let count = accesses
            .get(&memory_id)
            .copied()
            .unwrap_or_else(|| self.working_memory.get_access_count(memory_id, session_id));

        if count == 0 {
            return 1.0;
        }

        let raw_boost = (count as f64).ln().max(0.0) * 0.8;
        (1.0 + raw_boost).min(self.working_memory.config.max_boost)
    }

    fn compute_minerva2_activation(&self, similarity: f64) -> f64 {
        similarity.powi(3)
    }

    fn compute_base_level(&self, memory: &Memory, current_time: Timestamp) -> f64 {
        memory.metadata.base_level_activation(current_time)
    }

    fn compute_spreading(
        &self,
        memory: &Memory,
        graph: &AssociationGraph,
        all_memories: &[Memory],
        raw_similarities: &[f64],
        id_to_idx: &HashMap<Id, usize>,
    ) -> f64 {
        let mut total_spreading = 0.0;

        let mut visited = std::collections::HashSet::new();
        let mut current_frontier = vec![memory.id];
        visited.insert(memory.id);

        for depth in 0..self.spreading_depth {
            if current_frontier.is_empty() {
                break;
            }

            let depth_decay = self.spreading_decay.powi(depth as i32);
            let mut next_frontier = Vec::new();

            for &node_id in &current_frontier {
                let incoming_edges = graph.get_edges_to(node_id);
                for edge in &incoming_edges {
                    if visited.contains(&edge.from) {
                        continue;
                    }

                    if let Some(&idx) = id_to_idx.get(&edge.from) {
                        let source = &all_memories[idx];
                        let source_fan = graph.get_edges_from(source.id).len().max(1) as f64;
                        let fan_normalization = source_fan.sqrt();

                        let link_weight = match edge.kind {
                            LinkKind::Semantic => self.semantic_link_weight,
                            LinkKind::Episodic => self.episodic_link_weight,
                            LinkKind::Temporal => self.temporal_link_weight,
                            LinkKind::Causal => 0.8,
                        };

                        let similarity = raw_similarities[idx];
                        let activation = self.compute_minerva2_activation(similarity);
                        let contribution = activation * edge.strength * link_weight * depth_decay
                            / fan_normalization;
                        total_spreading += contribution;

                        visited.insert(edge.from);
                        next_frontier.push(edge.from);
                    }
                }

                if self.bidirectional_spreading {
                    let outgoing_edges = graph.get_edges_from(node_id);
                    for edge in &outgoing_edges {
                        if visited.contains(&edge.to) {
                            continue;
                        }

                        if let Some(&idx) = id_to_idx.get(&edge.to) {
                            let target = &all_memories[idx];
                            let target_fan = graph.get_edges_from(target.id).len().max(1) as f64;
                            let fan_normalization = (target_fan + 1.0).ln().max(1.0);

                            let link_weight = match edge.kind {
                                LinkKind::Semantic => self.semantic_link_weight * 0.7,
                                LinkKind::Episodic => self.episodic_link_weight * 0.7,
                                LinkKind::Temporal => self.temporal_link_weight * 0.7,
                                LinkKind::Causal => 0.6,
                            };

                            let similarity = raw_similarities[idx];
                            let activation = self.compute_minerva2_activation(similarity);
                            total_spreading +=
                                activation * edge.strength * link_weight * depth_decay
                                    / fan_normalization;

                            visited.insert(edge.to);
                            next_frontier.push(edge.to);
                        }
                    }
                }
            }

            current_frontier = next_frontier;
            current_frontier.sort_by(|a, b| a.0.cmp(&b.0));
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

    fn compute_temporal(
        &self,
        memory: &Memory,
        current_time: Timestamp,
        context: &RetrievalContext,
        raw_similarities: &[f64],
        id_to_idx: &std::collections::HashMap<Id, usize>,
    ) -> f64 {
        let mut temporal_score = 0.0;

        if let Some(ref temporal_ctx) = context.temporal_context {
            let tcm_similarity = temporal_ctx.similarity_to_embedding(&memory.embedding);

            let time_since_created = current_time.saturating_sub(memory.metadata.created_at) as f64;
            let recency_factor = (-memory.metadata.decay_rate * time_since_created / 1000.0).exp();

            temporal_score = tcm_similarity * 0.6 + recency_factor * 0.4;

            for link in &memory.temporal_links {
                let source_similarity = id_to_idx
                    .get(&link.source_id)
                    .and_then(|&idx| raw_similarities.get(idx).copied())
                    .unwrap_or(0.0);

                if source_similarity < 0.3 {
                    continue;
                }

                temporal_score += link.forward_strength * source_similarity * 0.15;
            }
        }

        if temporal_score == 0.0 {
            let time_since_access =
                current_time.saturating_sub(memory.metadata.last_accessed_at) as f64;
            temporal_score = (-memory.metadata.decay_rate * time_since_access / 1000.0).exp();
        }

        let session_decay = memory.metadata.compute_session_decay_rate(current_time);
        temporal_score * (1.0 - session_decay * 0.3)
    }

    fn compute_total_score(&self, scores: &RetrievalScores) -> f64 {
        let normalized_spreading = scores.spreading.tanh();
        let normalized_contextual = scores.contextual.clamp(0.0, 1.0);
        let normalized_temporal = scores.temporal.clamp(0.0, 1.0);

        let multiplicative_base = combine_activations_multiplicative(
            scores.similarity,
            scores.base_level,
            normalized_spreading * self.spreading_weight,
            scores.emotional,
        );

        let context_boost = 1.0 + (normalized_contextual * self.contextual_weight * 0.5);
        let temporal_boost = 1.0 + (normalized_temporal * self.temporal_weight * 0.3);

        let total = multiplicative_base * context_boost * temporal_boost;

        total.max(0.0)
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
        use std::collections::HashMap;

        if memories.is_empty() {
            return Vec::new();
        }

        let embeddings: Vec<&[f64]> = memories.iter().map(|m| m.embedding.as_slice()).collect();
        let raw_similarities =
            crate::recall::cosine_similarity_batch(&context.query_embedding, &embeddings);

        let id_to_idx: HashMap<Id, usize> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id, i))
            .collect();

        let mut matches: Vec<RetrievalMatch> = memories
            .iter()
            .enumerate()
            .map(|(idx, memory)| {
                let raw_similarity = raw_similarities[idx];
                let working_memory_boost = self.compute_wm_boost(
                    memory.id,
                    &context.working_memory_accesses,
                    context.session_id.as_deref(),
                );
                let adjusted_similarity = (raw_similarity * working_memory_boost).min(1.0);
                let similarity = self.compute_minerva2_activation(adjusted_similarity);
                let base_level = self.compute_base_level(memory, context.current_time);
                let spreading =
                    self.compute_spreading(memory, graph, memories, &raw_similarities, &id_to_idx);
                let emotional = self.compute_emotional(memory);
                let contextual = self.compute_contextual(memory, context);
                let temporal = self.compute_temporal(
                    memory,
                    context.current_time,
                    context,
                    &raw_similarities,
                    &id_to_idx,
                );
                let scores = RetrievalScores {
                    similarity,
                    raw_similarity,
                    base_level,
                    spreading,
                    emotional,
                    contextual,
                    temporal,
                    working_memory_boost,
                    total: 0.0,
                };

                let total = self.compute_total_score(&scores);

                RetrievalMatch {
                    memory_id: memory.id,
                    scores: RetrievalScores { total, ..scores },
                    probability: 0.0,
                }
            })
            .collect();

        let activation_threshold = 0.3;
        let noise_parameter = 0.1;

        let scores: Vec<f64> = matches.iter().map(|m| m.score()).collect();
        let probabilities = crate::recall::retrieval_probability_batch(
            &scores,
            activation_threshold,
            noise_parameter,
        );

        for (m, &prob) in matches.iter_mut().zip(probabilities.iter()) {
            m.probability = prob;
        }

        matches.retain(|m| m.probability >= context.min_threshold);

        matches.sort_by(|a, b| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.memory_id.0.cmp(&b.memory_id.0))
        });

        if matches.len() > context.max_results {
            matches.truncate(context.max_results);
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
    use crate::memory::{AssociationEdge, MemoryBuilder};
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

        let memories = vec![
            central_mem.clone(),
            peripheral1.clone(),
            peripheral2.clone(),
        ];
        let mut graph = AssociationGraph::new();

        graph.add_edge(AssociationEdge::new(
            central_mem.id,
            peripheral1.id,
            0.8,
            crate::types::LinkKind::Semantic,
            current_time,
        ));
        graph.add_edge(AssociationEdge::new(
            central_mem.id,
            peripheral2.id,
            0.8,
            crate::types::LinkKind::Semantic,
            current_time,
        ));
        graph.add_edge(AssociationEdge::new(
            peripheral1.id,
            peripheral2.id,
            0.5,
            crate::types::LinkKind::Semantic,
            current_time,
        ));

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
    async fn test_minerva2_activation_bounded() {
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

    #[test]
    fn test_working_memory_boost() {
        let wm = WorkingMemoryBoost::default();
        let mem_id = Id::new();
        let session = "test-session";

        assert_eq!(wm.compute_boost(mem_id, Some(session)), 0.0);

        wm.record_access(mem_id, session);
        wm.record_access(mem_id, session);
        let boost1 = wm.compute_boost(mem_id, Some(session));
        assert!(
            boost1 >= 1.0,
            "multiple accesses should give boost >= 1.0, got {}",
            boost1
        );

        wm.record_access(mem_id, session);
        wm.record_access(mem_id, session);
        let boost2 = wm.compute_boost(mem_id, Some(session));
        assert!(
            boost2 > boost1,
            "more accesses should increase boost: {} > {}",
            boost2,
            boost1
        );

        assert_eq!(wm.compute_boost(mem_id, None), 0.0);
        assert_eq!(wm.compute_boost(mem_id, Some("other-session")), 0.0);
    }

    #[tokio::test]
    async fn test_bidirectional_spreading() {
        let strategy = CognitiveRetrieval::new().with_bidirectional_spreading(true);
        let current_time = now();

        let mem_a = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time).build();
        let mem_b = MemoryBuilder::new(make_embedding(&[0.9, 0.1, 0.0]), current_time).build();

        let memories = vec![mem_a.clone(), mem_b.clone()];
        let mut graph = AssociationGraph::new();
        graph.add_edge(AssociationEdge::new(
            mem_a.id,
            mem_b.id,
            0.9,
            LinkKind::Semantic,
            current_time,
        ));

        let context = RetrievalContext::new(make_embedding(&[1.0, 0.0, 0.0]), current_time);

        let results = strategy.retrieve(&memories, &graph, &context).await;

        let mem_b_result = results.iter().find(|r| r.memory_id == mem_b.id);
        assert!(mem_b_result.is_some());

        if let Some(result) = mem_b_result {
            assert!(
                result.scores.spreading > 0.0,
                "bidirectional should give spreading to mem_b"
            );
        }
    }

    #[tokio::test]
    async fn test_session_working_memory_integration() {
        let strategy = CognitiveRetrieval::new();
        let current_time = now();

        let mem1 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("test memory")
            .build();
        let mem2 = MemoryBuilder::new(make_embedding(&[0.0, 1.0, 0.0]), current_time)
            .text("other memory")
            .build();

        let memories = vec![mem1.clone(), mem2.clone()];
        let graph = AssociationGraph::new();
        let session_id = "test-session";

        strategy.working_memory().record_access(mem1.id, session_id);
        strategy.working_memory().record_access(mem1.id, session_id);
        strategy.working_memory().record_access(mem1.id, session_id);

        let context = RetrievalContext::new(make_embedding(&[0.7, 0.3, 0.0]), current_time)
            .with_session_id(session_id);

        let results = strategy.retrieve(&memories, &graph, &context).await;

        let mem1_result = results.iter().find(|r| r.memory_id == mem1.id);
        let mem2_result = results.iter().find(|r| r.memory_id == mem2.id);

        if let (Some(r1), Some(r2)) = (mem1_result, mem2_result) {
            assert!(
                r1.scores.working_memory_boost > r2.scores.working_memory_boost,
                "frequently accessed memory should have higher WM boost"
            );
        }
    }
}
