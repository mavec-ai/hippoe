use serde::{Deserialize, Serialize};

use crate::recall::scorer::cosine_similarity;
use crate::types::{Embedding, Emotion, Id, LinkKind, Timestamp};

const MAX_ACCESS_TIMESTAMPS: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub context_vector: Embedding,
    pub drift_rate: f64,
    pub integration_rate: f64,
}

impl TemporalContext {
    pub fn new(dimensions: usize) -> Self {
        Self {
            context_vector: vec![0.0; dimensions],
            drift_rate: 0.9,
            integration_rate: 0.1,
        }
    }

    pub fn from_embedding(embedding: &[f64]) -> Self {
        Self {
            context_vector: embedding.to_vec(),
            drift_rate: 0.9,
            integration_rate: 0.1,
        }
    }

    pub fn update(&mut self, item_embedding: &[f64]) {
        if item_embedding.len() != self.context_vector.len() {
            return;
        }

        for (ctx, item) in self.context_vector.iter_mut().zip(item_embedding.iter()) {
            *ctx = self.drift_rate * *ctx + self.integration_rate * *item;
        }

        let norm: f64 = self
            .context_vector
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for x in &mut self.context_vector {
                *x /= norm;
            }
        } else {
            for x in &mut self.context_vector {
                *x = 0.0;
            }
        }
    }

    pub fn similarity(&self, other_context: &TemporalContext) -> f64 {
        cosine_similarity(&self.context_vector, &other_context.context_vector).max(0.0)
    }

    pub fn similarity_to_embedding(&self, embedding: &[f64]) -> f64 {
        cosine_similarity(&self.context_vector, embedding).max(0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Id,
    pub content: MemoryContent,
    pub embedding: Embedding,
    pub metadata: MemoryMetadata,
    pub associations: Vec<Association>,
    pub temporal_links: Vec<TemporalLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContent {
    pub text: Option<String>,
    pub structured: Option<serde_json::Value>,
    pub raw: Option<Vec<u8>>,
}

impl MemoryContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            structured: None,
            raw: None,
        }
    }

    pub fn structured(data: serde_json::Value) -> Self {
        Self {
            text: None,
            structured: Some(data),
            raw: None,
        }
    }

    pub fn raw(data: Vec<u8>) -> Self {
        Self {
            text: None,
            structured: None,
            raw: Some(data),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_none() && self.structured.is_none() && self.raw.is_none()
    }

    pub fn to_string(&self) -> Option<String> {
        self.text
            .clone()
            .or_else(|| self.structured.as_ref().map(|s| s.to_string()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationState {
    Fresh,
    Consolidating,
    Consolidated,
    Reconsolidating,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub last_accessed_at: Timestamp,
    pub access_count: u64,
    pub access_timestamps: Vec<Timestamp>,
    pub emotional_weight: Emotion,
    pub decay_rate: f64,
    pub importance: f64,
    pub context: Option<String>,
    pub tags: Vec<String>,
    pub lability: f64,
    pub consolidation_threshold: f64,
    pub consolidation_state: ConsolidationState,
    pub last_consolidation_at: Timestamp,
}

impl MemoryMetadata {
    pub fn new(created_at: Timestamp) -> Self {
        Self {
            created_at,
            updated_at: created_at,
            last_accessed_at: created_at,
            access_count: 0,
            access_timestamps: vec![created_at],
            emotional_weight: Emotion::default(),
            decay_rate: 0.1,
            importance: 0.5,
            context: None,
            tags: Vec::new(),
            lability: 1.0,
            consolidation_threshold: 0.3,
            consolidation_state: ConsolidationState::Fresh,
            last_consolidation_at: created_at,
        }
    }

    pub fn accessed(&mut self, at: Timestamp) {
        self.access_count += 1;
        self.last_accessed_at = at;
        self.access_timestamps.push(at);
        if self.access_timestamps.len() > MAX_ACCESS_TIMESTAMPS {
            self.access_timestamps.remove(0);
        }
    }

    pub fn with_emotion(mut self, valence: f64, arousal: f64) -> Self {
        self.emotional_weight = Emotion::new(valence, arousal);
        self
    }

    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    pub fn remove_tag(&mut self, tag: &str) {
        self.tags.retain(|t| t != tag);
    }

    pub fn base_level_activation(&self, current_time: Timestamp) -> f64 {
        let time_since_creation =
            (current_time.saturating_sub(self.created_at) as f64 / 1000.0).max(1.0);

        let access_boost = (self.access_count as f64 + 1.0).ln();
        let time_penalty = self.decay_rate * time_since_creation.ln();
        let base = access_boost - time_penalty;

        (base * self.importance).max(0.0)
    }

    pub fn compute_surprise(&self, expected_embedding: &[f64], actual_embedding: &[f64]) -> f64 {
        let similarity = cosine_similarity(expected_embedding, actual_embedding).clamp(0.0, 1.0);
        1.0 - similarity
    }

    pub fn should_reconsolidate(&self, surprise: f64) -> bool {
        surprise > self.consolidation_threshold
    }

    pub fn compute_session_decay_rate(&self, current_time: Timestamp) -> f64 {
        let hours_ago = (current_time.saturating_sub(self.last_accessed_at) as f64) / 3_600_000.0;

        if hours_ago < 0.0 {
            return 0.5;
        }

        if hours_ago < 0.5 {
            0.3
        } else if hours_ago < 2.0 {
            0.4
        } else if hours_ago < 24.0 {
            0.45
        } else {
            0.5
        }
    }

    pub fn update_consolidation_state(&mut self, current_time: Timestamp) {
        let hours_since_creation =
            (current_time.saturating_sub(self.created_at) as f64) / 3_600_000.0;
        let hours_since_consolidation =
            (current_time.saturating_sub(self.last_consolidation_at) as f64) / 3_600_000.0;

        self.consolidation_state = match self.consolidation_state {
            ConsolidationState::Fresh => {
                if hours_since_creation > 1.0 {
                    ConsolidationState::Consolidating
                } else {
                    ConsolidationState::Fresh
                }
            }
            ConsolidationState::Consolidating => {
                if hours_since_creation > 24.0 {
                    ConsolidationState::Consolidated
                } else {
                    ConsolidationState::Consolidating
                }
            }
            ConsolidationState::Consolidated => {
                if hours_since_consolidation < 2.0 {
                    ConsolidationState::Reconsolidating
                } else {
                    ConsolidationState::Consolidated
                }
            }
            ConsolidationState::Reconsolidating => {
                if hours_since_consolidation > 2.0 {
                    ConsolidationState::Consolidated
                } else {
                    ConsolidationState::Reconsolidating
                }
            }
        };
    }

    pub fn apply_reconsolidation(&mut self) {
        self.lability *= 0.9;
        self.lability = self.lability.max(0.1);
    }

    pub fn apply_reconsolidation_with_surprise(&mut self, surprise: f64) {
        if surprise > 0.8 {
            self.lability = (self.lability + 0.5).min(1.0);
        }
        self.lability *= 0.9;
        self.lability = self.lability.max(0.1);
    }

    pub fn emotional_modulation(&self) -> f64 {
        self.emotional_weight.weight() * self.importance
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalLink {
    pub source_id: Id,
    pub target_id: Id,
    pub forward_strength: f64,
    pub backward_strength: f64,
    pub temporal_distance: usize,
    pub created_at: Timestamp,
}

impl TemporalLink {
    pub fn new(source_id: Id, target_id: Id, distance: usize, created_at: Timestamp) -> Self {
        let forward_strength = 0.8_f64.powi(distance as i32);
        let backward_strength = forward_strength * 0.6;

        Self {
            source_id,
            target_id,
            forward_strength,
            backward_strength,
            temporal_distance: distance,
            created_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    pub target_id: Id,
    pub strength: f64,
    pub kind: LinkKind,
    pub created_at: Timestamp,
    pub last_activated: Timestamp,
    pub last_decayed_at: Timestamp,
    pub activation_count: u64,
}

impl Association {
    pub fn new(target_id: Id, strength: f64, kind: LinkKind, created_at: Timestamp) -> Self {
        Self {
            target_id,
            strength: strength.clamp(0.0, 1.0),
            kind,
            created_at,
            last_activated: created_at,
            last_decayed_at: created_at,
            activation_count: 0,
        }
    }

    pub fn semantic(target_id: Id, strength: f64, created_at: Timestamp) -> Self {
        Self::new(target_id, strength, LinkKind::Semantic, created_at)
    }

    pub fn episodic(target_id: Id, strength: f64, created_at: Timestamp) -> Self {
        Self::new(target_id, strength, LinkKind::Episodic, created_at)
    }

    pub fn causal(target_id: Id, strength: f64, created_at: Timestamp) -> Self {
        Self::new(target_id, strength, LinkKind::Causal, created_at)
    }

    pub fn temporal(target_id: Id, strength: f64, created_at: Timestamp) -> Self {
        Self::new(target_id, strength, LinkKind::Temporal, created_at)
    }

    pub fn activate(&mut self, at: Timestamp) {
        self.last_activated = at;
        self.activation_count += 1;
    }

    pub fn decay(&mut self, current_time: Timestamp, decay_rate: f64) {
        let time_since_last_decay =
            current_time.saturating_sub(self.last_decayed_at) as f64 / 1000.0;
        if time_since_last_decay > 0.0 {
            let decay_factor = (-decay_rate * time_since_last_decay).exp();
            self.strength *= decay_factor;
            self.last_decayed_at = current_time;
        }
    }
}

impl Memory {
    pub fn new(content: MemoryContent, embedding: Embedding, created_at: Timestamp) -> Self {
        Self {
            id: Id::new(),
            content,
            embedding,
            metadata: MemoryMetadata::new(created_at),
            associations: Vec::new(),
            temporal_links: Vec::new(),
        }
    }

    pub fn text(text: impl Into<String>, embedding: Embedding, created_at: Timestamp) -> Self {
        Self::new(MemoryContent::text(text), embedding, created_at)
    }

    pub fn structured(
        data: serde_json::Value,
        embedding: Embedding,
        created_at: Timestamp,
    ) -> Self {
        Self::new(MemoryContent::structured(data), embedding, created_at)
    }

    pub fn accessed(&mut self, at: Timestamp) {
        self.metadata.accessed(at);
    }

    pub fn add_association(&mut self, association: Association) {
        if let Some(existing) = self
            .associations
            .iter_mut()
            .find(|a| a.target_id == association.target_id && a.kind == association.kind)
        {
            existing.strength = (existing.strength + association.strength).min(1.0);
            existing.activate(association.last_activated);
        } else {
            self.associations.push(association);
        }
    }

    pub fn remove_association(&mut self, target_id: Id, kind: LinkKind) {
        self.associations
            .retain(|a| !(a.target_id == target_id && a.kind == kind));
    }

    pub fn get_association(&self, target_id: Id, kind: LinkKind) -> Option<&Association> {
        self.associations
            .iter()
            .find(|a| a.target_id == target_id && a.kind == kind)
    }

    pub fn get_associations_by_kind(&self, kind: LinkKind) -> Vec<&Association> {
        self.associations
            .iter()
            .filter(|a| a.kind == kind)
            .collect()
    }

    pub fn association_strength(&self, target_id: Id) -> f64 {
        self.associations
            .iter()
            .filter(|a| a.target_id == target_id)
            .map(|a| a.strength)
            .sum()
    }

    pub fn decay_associations(&mut self, current_time: Timestamp) {
        for association in &mut self.associations {
            association.decay(current_time, self.metadata.decay_rate);
        }
        self.associations.retain(|a| a.strength > 0.01);
    }

    pub fn spreading_activation_potential(&self) -> f64 {
        let base_activation = self
            .metadata
            .base_level_activation(self.metadata.last_accessed_at);
        let emotional_boost = self.metadata.emotional_modulation();

        (base_activation + emotional_boost) / 2.0
    }

    pub fn reconsolidate(
        &mut self,
        new_embedding: &[f64],
        learning_rate: f64,
        current_time: Timestamp,
        surprise: f64,
    ) {
        let alpha = learning_rate * self.metadata.lability;

        if self.embedding.len() == new_embedding.len() {
            for (emb, new) in self.embedding.iter_mut().zip(new_embedding.iter()) {
                *emb = (1.0 - alpha) * *emb + alpha * new;
            }

            let norm: f64 = self.embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut self.embedding {
                    *x /= norm;
                }
            } else {
                for x in &mut self.embedding {
                    *x = 0.0;
                }
            }
        }

        self.metadata.apply_reconsolidation_with_surprise(surprise);
        self.metadata.last_consolidation_at = current_time;
        self.metadata.consolidation_state = ConsolidationState::Reconsolidating;
    }
}

pub struct MemoryBuilder {
    id: Option<Id>,
    content: MemoryContent,
    embedding: Embedding,
    metadata: MemoryMetadata,
    associations: Vec<Association>,
    temporal_links: Vec<TemporalLink>,
}

impl MemoryBuilder {
    pub fn new(embedding: Embedding, created_at: Timestamp) -> Self {
        Self {
            id: None,
            content: MemoryContent::text(String::new()),
            embedding,
            metadata: MemoryMetadata::new(created_at),
            associations: Vec::new(),
            temporal_links: Vec::new(),
        }
    }

    pub fn id(mut self, id: Id) -> Self {
        self.id = Some(id);
        self
    }

    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.content = MemoryContent::text(text);
        self
    }

    pub fn structured(mut self, data: serde_json::Value) -> Self {
        self.content = MemoryContent::structured(data);
        self
    }

    pub fn raw(mut self, data: Vec<u8>) -> Self {
        self.content = MemoryContent::raw(data);
        self
    }

    pub fn emotion(mut self, valence: f64, arousal: f64) -> Self {
        self.metadata = self.metadata.with_emotion(valence, arousal);
        self
    }

    pub fn decay_rate(mut self, rate: f64) -> Self {
        self.metadata = self.metadata.with_decay_rate(rate);
        self
    }

    pub fn importance(mut self, importance: f64) -> Self {
        self.metadata = self.metadata.with_importance(importance);
        self
    }

    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.metadata = self.metadata.with_context(context);
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.metadata.add_tag(tag);
        self
    }

    pub fn association(mut self, association: Association) -> Self {
        self.associations.push(association);
        self
    }

    pub fn link(mut self, target_id: Id, strength: f64, kind: LinkKind, at: Timestamp) -> Self {
        self.associations
            .push(Association::new(target_id, strength, kind, at));
        self
    }

    pub fn build(self) -> Memory {
        Memory {
            id: self.id.unwrap_or_default(),
            content: self.content,
            embedding: self.embedding,
            metadata: self.metadata,
            associations: self.associations,
            temporal_links: self.temporal_links,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_memory_creation() {
        let embedding = make_embedding(&[1.0, 2.0, 3.0]);
        let now = now();

        let memory = Memory::text("test memory", embedding.clone(), now);

        assert!(memory.content.text.is_some());
        assert_eq!(memory.metadata.created_at, now);
        assert_eq!(memory.metadata.access_count, 0);
    }

    #[test]
    fn test_memory_access() {
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let now = now();

        let mut memory = Memory::text("test", embedding, now);
        memory.accessed(now + 1000);

        assert_eq!(memory.metadata.access_count, 1);
        assert_eq!(memory.metadata.last_accessed_at, now + 1000);
    }

    #[test]
    fn test_association() {
        let id2 = Id::new();
        let now = now();

        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let mut memory = Memory::text("test", embedding, now);

        let association = Association::semantic(id2, 0.8, now);
        memory.add_association(association);

        assert_eq!(memory.associations.len(), 1);
        assert_eq!(memory.association_strength(id2), 0.8);
    }

    #[test]
    fn test_metadata_decay() {
        let now = now();
        let mut metadata = MemoryMetadata::new(now)
            .with_decay_rate(0.5)
            .with_importance(0.8);

        metadata.accessed(now);

        let activation = metadata.base_level_activation(now + 2000);
        assert!(activation > 0.0);
    }

    #[test]
    fn test_memory_builder() {
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let now = now();
        let target_id = Id::new();

        let memory = MemoryBuilder::new(embedding.clone(), now)
            .text("test memory")
            .emotion(0.7, 0.8)
            .importance(0.9)
            .context("test context")
            .tag("important")
            .link(target_id, 0.5, LinkKind::Semantic, now)
            .build();

        assert!(memory.content.text.is_some());
        assert_eq!(memory.metadata.emotional_weight.valence, 0.7);
        assert_eq!(memory.metadata.importance, 0.9);
        assert_eq!(memory.metadata.context, Some("test context".to_string()));
        assert!(memory.metadata.tags.contains(&"important".to_string()));
        assert_eq!(memory.associations.len(), 1);
    }

    #[test]
    fn test_association_decay() {
        let target_id = Id::new();
        let now = now();

        let mut association = Association::semantic(target_id, 0.9, now);
        association.decay(now + 1000, 0.5);

        assert!(association.strength < 0.9);
        assert!(association.strength > 0.0);
    }

    #[test]
    fn test_temporal_context_creation() {
        let ctx = TemporalContext::new(3);
        assert_eq!(ctx.context_vector.len(), 3);
        assert_eq!(ctx.drift_rate, 0.9);
        assert_eq!(ctx.integration_rate, 0.1);
    }

    #[test]
    fn test_temporal_context_update() {
        let mut ctx = TemporalContext::new(3);

        ctx.update(&[1.0, 0.0, 0.0]);

        let norm: f64 = ctx.context_vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_temporal_context_similarity() {
        let mut ctx1 = TemporalContext::new(3);
        let mut ctx2 = TemporalContext::new(3);

        ctx1.update(&[1.0, 0.0, 0.0]);
        ctx2.update(&[1.0, 0.0, 0.0]);

        let similarity = ctx1.similarity(&ctx2);
        assert!(similarity > 0.9);
    }

    #[test]
    fn test_temporal_context_drift() {
        let mut ctx = TemporalContext::new(3);

        ctx.update(&[1.0, 0.0, 0.0]);
        let state1 = ctx.context_vector.clone();

        ctx.update(&[0.0, 1.0, 0.0]);
        let state2 = ctx.context_vector.clone();

        let changed = state1
            .iter()
            .zip(state2.iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        assert!(changed);
    }

    #[test]
    fn test_temporal_context_similarity_to_embedding() {
        let mut ctx = TemporalContext::new(3);
        ctx.update(&[1.0, 0.0, 0.0]);

        let similarity = ctx.similarity_to_embedding(&[1.0, 0.0, 0.0]);
        assert!(similarity > 0.9);

        let low_similarity = ctx.similarity_to_embedding(&[0.0, 1.0, 0.0]);
        assert!(low_similarity < similarity);
    }

    #[test]
    fn test_reconsolidation_trigger() {
        let metadata = MemoryMetadata::new(now());

        assert!(!metadata.should_reconsolidate(0.1));
        assert!(metadata.should_reconsolidate(0.8));
    }

    #[test]
    fn test_reconsolidation_lability_decay() {
        let mut metadata = MemoryMetadata::new(now());
        let initial_lability = metadata.lability;

        metadata.apply_reconsolidation();

        assert!(metadata.lability < initial_lability);
    }

    #[test]
    fn test_memory_reconsolidate() {
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let now = now();
        let mut memory = Memory::text("test", embedding.clone(), now);

        let initial_embedding = memory.embedding.clone();
        memory.reconsolidate(&[0.9, 0.1, 0.0], 0.5, now, 0.5);

        let changed = initial_embedding
            .iter()
            .zip(memory.embedding.iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        assert!(changed);

        let norm: f64 = memory.embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_reconsolidation_stability() {
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let now = now();
        let mut memory = Memory::text("test", embedding.clone(), now);

        for _ in 0..10 {
            memory.reconsolidate(&[1.0, 0.0, 0.0], 0.1, now, 0.5);
        }

        let norm: f64 = memory.embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
        assert!(memory.embedding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_reconsolidation_with_small_learning_rate() {
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let now = now();
        let mut memory = Memory::text("test", embedding.clone(), now);

        let initial_embedding = memory.embedding.clone();
        memory.reconsolidate(&[0.0, 1.0, 0.0], 0.01, now, 0.5);

        let change: f64 = initial_embedding
            .iter()
            .zip(memory.embedding.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(change < 0.1);
    }

    #[test]
    fn test_consolidation_state_transitions() {
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let now = now();
        let mut memory = Memory::text("test", embedding.clone(), now);

        assert_eq!(
            memory.metadata.consolidation_state,
            ConsolidationState::Fresh
        );

        let two_hours_later = now + 2 * 3_600_000;
        memory.metadata.update_consolidation_state(two_hours_later);
        assert_eq!(
            memory.metadata.consolidation_state,
            ConsolidationState::Consolidating
        );

        let twenty_six_hours_later = now + 26 * 3_600_000;
        memory
            .metadata
            .update_consolidation_state(twenty_six_hours_later);
        assert_eq!(
            memory.metadata.consolidation_state,
            ConsolidationState::Consolidated
        );

        memory.metadata.last_consolidation_at = twenty_six_hours_later;
        let one_hour_after = twenty_six_hours_later + 3_600_000;
        memory.metadata.update_consolidation_state(one_hour_after);
        assert_eq!(
            memory.metadata.consolidation_state,
            ConsolidationState::Reconsolidating
        );
    }

    #[test]
    fn test_session_decay_rate() {
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let now = now();
        let mut memory = Memory::text("test", embedding.clone(), now);

        memory.metadata.last_accessed_at = now;
        let decay_recent = memory.metadata.compute_session_decay_rate(now);
        assert!(decay_recent < 0.4);

        memory.metadata.last_accessed_at = now - 3_600_000;
        let decay_one_hour = memory.metadata.compute_session_decay_rate(now);
        assert!((0.4..0.45).contains(&decay_one_hour));

        memory.metadata.last_accessed_at = now - 25 * 3_600_000;
        let decay_one_day = memory.metadata.compute_session_decay_rate(now);
        assert!(decay_one_day >= 0.5);
    }

    #[test]
    fn test_temporal_link_asymmetric_strength() {
        let source_id = Id::new();
        let target_id = Id::new();
        let now = now();

        let link_distance_1 = TemporalLink::new(source_id, target_id, 1, now);
        assert!(link_distance_1.forward_strength > link_distance_1.backward_strength);
        assert!((link_distance_1.forward_strength - 0.8).abs() < 0.001);
        assert!((link_distance_1.backward_strength - 0.48).abs() < 0.01);

        let link_distance_3 = TemporalLink::new(source_id, target_id, 3, now);
        assert!(link_distance_3.forward_strength < link_distance_1.forward_strength);
    }

    #[test]
    fn test_memory_temporal_links() {
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);
        let now = now();
        let mut memory = Memory::text("test", embedding.clone(), now);

        assert!(memory.temporal_links.is_empty());

        let target_id = Id::new();
        let link = TemporalLink::new(memory.id, target_id, 1, now);
        memory.temporal_links.push(link);

        assert_eq!(memory.temporal_links.len(), 1);
        assert_eq!(memory.temporal_links[0].source_id, memory.id);
        assert_eq!(memory.temporal_links[0].target_id, target_id);
    }
}
