use crate::config::Config;
use crate::error::Result;
use crate::memory::{
    AssociationBuilder, AssociationEdge, AssociationGraph, Memory, TemporalContext,
    compute_association_strength,
};
use crate::recall::{CognitiveRetrieval, RetrievalContext, RetrievalMatch, RetrievalStrategy};
use crate::storage::Storage;
use crate::types::{Embedding, Id, LinkKind, now};
use std::sync::{Arc, RwLock};

pub struct Hippocampus<S: Storage> {
    config: Config,
    storage: S,
    association_builder: AssociationBuilder,
    retrieval_strategy: Box<dyn RetrievalStrategy>,
    temporal_context: Arc<RwLock<TemporalContext>>,
}

impl<S: Storage> Hippocampus<S> {
    pub fn new(storage: S) -> Result<Self> {
        HippocampusBuilder::default().build(storage)
    }

    pub fn builder() -> HippocampusBuilder {
        HippocampusBuilder::default()
    }

    pub async fn memorize(&self, memory: Memory) -> Result<Id> {
        let id = memory.id;

        {
            let mut ctx = self.temporal_context.write().unwrap();
            ctx.update(&memory.embedding);
        }

        let existing = self.storage.all().await?;

        let mut graph = self.storage.get_graph().await?;

        self.association_builder
            .build_associations(&memory, &existing, &mut graph);

        self.storage.update_graph(|g| *g = graph).await?;

        self.storage.put(memory).await?;
        Ok(id)
    }

    pub async fn memorize_batch(&self, memories: Vec<Memory>) -> Result<Vec<Id>> {
        let existing = self.storage.all().await?;
        let mut graph = self.storage.get_graph().await?;

        for memory in &memories {
            self.association_builder
                .build_associations(memory, &existing, &mut graph);
        }

        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                let strength = compute_association_strength(&memories[i], &memories[j]);
                if strength > 0.5 {
                    self.association_builder.build_bidirectional(
                        &memories[i],
                        &memories[j],
                        strength,
                        LinkKind::Semantic,
                        &mut graph,
                    );
                }
            }
        }

        self.storage.update_graph(|g| *g = graph).await?;

        let mut ids = Vec::with_capacity(memories.len());
        for memory in memories {
            let id = memory.id;
            self.storage.put(memory).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    pub async fn recall(&self, probe: Embedding) -> Result<Vec<RetrievalMatch>> {
        {
            let mut ctx = self.temporal_context.write().unwrap();
            ctx.update(&probe);
        }

        let memories = self.storage.all().await?;
        let graph = self.storage.get_graph().await?;

        let temporal_ctx = {
            let ctx = self.temporal_context.read().unwrap();
            ctx.clone()
        };

        let context = RetrievalContext::new(probe.clone(), now())
            .with_min_threshold(self.config.min_score)
            .with_max_results(self.config.max_results)
            .with_temporal_context(temporal_ctx);

        let matches = self
            .retrieval_strategy
            .retrieve(&memories, &graph, &context)
            .await;

        self.apply_reconsolidation(&matches, &probe).await?;

        Ok(matches)
    }

    pub async fn recall_with_strategy(
        &self,
        probe: Embedding,
        strategy: &dyn RetrievalStrategy,
    ) -> Result<Vec<RetrievalMatch>> {
        let memories = self.storage.all().await?;
        let graph = self.storage.get_graph().await?;

        let context = RetrievalContext::new(probe.clone(), now())
            .with_min_threshold(self.config.min_score)
            .with_max_results(self.config.max_results);

        let matches = strategy.retrieve(&memories, &graph, &context).await;

        self.apply_reconsolidation(&matches, &probe).await?;

        Ok(matches)
    }

    pub async fn recall_by_tag(&self, tag: &str) -> Result<Vec<Memory>> {
        self.storage.find_by_tag(tag).await
    }

    pub async fn recall_by_context(&self, context: &str) -> Result<Vec<Memory>> {
        self.storage.find_by_context(context).await
    }

    pub async fn recall_associated(&self, id: Id, max_depth: usize) -> Result<Vec<Memory>> {
        self.storage.get_associated(id, max_depth).await
    }

    pub async fn forget(&self, id: Id) -> Result<()> {
        self.storage.remove(id).await
    }

    pub async fn get(&self, id: Id) -> Result<Option<Memory>> {
        let memory = self.storage.get(id).await?;

        if let Some(mut mem) = memory {
            mem.metadata.accessed(now());
            self.storage.put(mem.clone()).await?;
            Ok(Some(mem))
        } else {
            Ok(None)
        }
    }

    pub async fn all(&self) -> Result<Vec<Memory>> {
        self.storage.all().await
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub async fn get_graph(&self) -> Result<AssociationGraph> {
        self.storage.get_graph().await
    }

    pub async fn strengthen_association(
        &self,
        from: Id,
        to: Id,
        kind: LinkKind,
        increment: f64,
    ) -> Result<()> {
        self.storage
            .update_graph(|g| {
                if let Some(edge) = g.get_edge(from, to, kind) {
                    let new_strength = (edge.strength + increment).min(1.0);
                    let new_edge = AssociationEdge::new(
                        from,
                        to,
                        new_strength,
                        kind,
                        now(),
                    );
                    g.add_edge(new_edge);
                }
            })
            .await
    }

    pub async fn create_association(
        &self,
        from: Id,
        to: Id,
        strength: f64,
        kind: LinkKind,
    ) -> Result<()> {
        self.storage
            .update_graph(|g| {
                let edge = AssociationEdge::new(from, to, strength, kind, now());
                g.add_edge(edge);
            })
            .await
    }

    async fn apply_reconsolidation(&self, matches: &[RetrievalMatch], probe: &Embedding) -> Result<()> {
        for m in matches.iter().take(5) {
            if let Some(mut memory) = self.storage.get(m.memory_id).await? {
                let surprise = 1.0 - m.scores.similarity;
                
                if memory.metadata.should_reconsolidate(surprise) {
                    memory.reconsolidate(probe, 0.1);
                }
                
                memory.metadata.accessed(now());
                memory.metadata.importance = (memory.metadata.importance * 1.05).min(1.0);
                self.storage.put(memory).await?;
            }
        }
        Ok(())
    }

    pub fn query(&self) -> crate::recall::MemoryQueryBuilder<'_, S> {
        crate::recall::MemoryQueryBuilder::new(self)
    }

    pub(crate) async fn retrieve_with_default_strategy(
        &self,
        memories: &[Memory],
        graph: &AssociationGraph,
        context: &RetrievalContext,
    ) -> Vec<RetrievalMatch> {
        self.retrieval_strategy.retrieve(memories, graph, context).await
    }
}

#[derive(Default)]
pub struct HippocampusBuilder {
    decay_rate: Option<f64>,
    min_score: Option<f64>,
    max_results: Option<usize>,
    emotion_weight: Option<f64>,
    context_weight: Option<f64>,
    semantic_threshold: Option<f64>,
    episodic_threshold: Option<f64>,
    temporal_threshold: Option<f64>,
    max_associations: Option<usize>,
}

impl HippocampusBuilder {
    pub fn decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = Some(rate);
        self
    }

    pub fn min_score(mut self, score: f64) -> Self {
        self.min_score = Some(score);
        self
    }

    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = Some(max);
        self
    }

    pub fn emotion_weight(mut self, weight: f64) -> Self {
        self.emotion_weight = Some(weight);
        self
    }

    pub fn context_weight(mut self, weight: f64) -> Self {
        self.context_weight = Some(weight);
        self
    }

    pub fn semantic_threshold(mut self, threshold: f64) -> Self {
        self.semantic_threshold = Some(threshold);
        self
    }

    pub fn episodic_threshold(mut self, threshold: f64) -> Self {
        self.episodic_threshold = Some(threshold);
        self
    }

    pub fn temporal_threshold(mut self, threshold: f64) -> Self {
        self.temporal_threshold = Some(threshold);
        self
    }

    pub fn max_associations(mut self, max: usize) -> Self {
        self.max_associations = Some(max);
        self
    }

    pub fn build<S: Storage>(self, storage: S) -> Result<Hippocampus<S>> {
        let mut config_builder = Config::builder();

        if let Some(v) = self.decay_rate {
            config_builder = config_builder.decay_rate(v);
        }
        if let Some(v) = self.min_score {
            config_builder = config_builder.min_score(v);
        }
        if let Some(v) = self.max_results {
            config_builder = config_builder.max_results(v);
        }
        if let Some(v) = self.emotion_weight {
            config_builder = config_builder.emotion_weight(v);
        }
        if let Some(v) = self.context_weight {
            config_builder = config_builder.context_weight(v);
        }

        let config = config_builder.build()?;

        let mut assoc_builder = AssociationBuilder::new();

        if let Some(v) = self.semantic_threshold {
            assoc_builder = assoc_builder.with_semantic_threshold(v);
        }
        if let Some(v) = self.episodic_threshold {
            assoc_builder = assoc_builder.with_episodic_threshold(v);
        }
        if let Some(v) = self.temporal_threshold {
            assoc_builder = assoc_builder.with_temporal_threshold(v);
        }
        if let Some(v) = self.max_associations {
            assoc_builder = assoc_builder.with_max_associations(v);
        }

        let mut retrieval_strategy = CognitiveRetrieval::new();
        if let Some(v) = self.emotion_weight {
            retrieval_strategy = retrieval_strategy.with_emotional_weight(v);
        }
        if let Some(v) = self.context_weight {
            retrieval_strategy = retrieval_strategy.with_contextual_weight(v);
        }

        let embedding_dim = 384;
        let temporal_context = Arc::new(RwLock::new(TemporalContext::new(embedding_dim)));

        Ok(Hippocampus {
            config,
            storage,
            association_builder: assoc_builder,
            retrieval_strategy: Box::new(retrieval_strategy),
            temporal_context,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBuilder;
    use crate::storage::InMemoryStorage;

    fn make_embedding(values: &[f64]) -> Embedding {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[tokio::test]
    async fn test_hippocampus_memorize_recall() {
        let storage = InMemoryStorage::new();
        let hippoe = Hippocampus::new(storage).unwrap();

        let mem1 = Memory::text("memory 1", make_embedding(&[1.0, 0.0, 0.0]), now());
        let mem2 = Memory::text("memory 2", make_embedding(&[0.9, 0.1, 0.0]), now());
        let mem3 = Memory::text("memory 3", make_embedding(&[0.1, 0.9, 0.0]), now());

        hippoe.memorize(mem1.clone()).await.unwrap();
        hippoe.memorize(mem2.clone()).await.unwrap();
        hippoe.memorize(mem3.clone()).await.unwrap();

        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let matches = hippoe.recall(probe).await.unwrap();

        assert!(matches.len() >= 2);
    }

    #[tokio::test]
    async fn test_hippocampus_auto_associations() {
        let storage = InMemoryStorage::new();
        let hippoe = Hippocampus::new(storage).unwrap();

        let mem1 = Memory::text("memory 1", make_embedding(&[1.0, 0.0, 0.0]), now());
        let mem2 = Memory::text("memory 2", make_embedding(&[0.95, 0.1, 0.0]), now());

        hippoe.memorize(mem1.clone()).await.unwrap();
        hippoe.memorize(mem2.clone()).await.unwrap();

        let graph = hippoe.get_graph().await.unwrap();
        let has_association = graph.has_edge(mem1.id, mem2.id) || graph.has_edge(mem2.id, mem1.id);
        assert!(has_association);
    }

    #[tokio::test]
    async fn test_hippocampus_recall_by_tag() {
        let storage = InMemoryStorage::new();
        let hippoe = Hippocampus::new(storage).unwrap();

        let mem = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), now())
            .text("important note")
            .tag("important")
            .build();

        hippoe.memorize(mem).await.unwrap();

        let results = hippoe.recall_by_tag("important").await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_hippocampus_recall_associated() {
        let storage = InMemoryStorage::new();
        let hippoe = Hippocampus::new(storage).unwrap();

        let mem1 = Memory::text("m1", make_embedding(&[1.0, 0.0, 0.0]), now());
        let mem2 = Memory::text("m2", make_embedding(&[0.95, 0.05, 0.0]), now());

        hippoe.memorize(mem1.clone()).await.unwrap();
        hippoe.memorize(mem2.clone()).await.unwrap();

        let graph = hippoe.get_graph().await.unwrap();
        let has_edge = graph.has_edge(mem1.id, mem2.id) || graph.has_edge(mem2.id, mem1.id);
        assert!(has_edge, "Expected association between memories");

        let associated = hippoe.recall_associated(mem1.id, 2).await.unwrap();
        assert!(!associated.is_empty());
    }

    #[tokio::test]
    async fn test_hippocampus_builder() {
        let storage = InMemoryStorage::new();
        let hippoe = HippocampusBuilder::default()
            .decay_rate(0.3)
            .min_score(0.05)
            .max_results(20)
            .semantic_threshold(0.8)
            .episodic_threshold(0.6)
            .build(storage)
            .unwrap();

        assert_eq!(hippoe.config().decay_rate, 0.3);
        assert_eq!(hippoe.config().min_score, 0.05);
    }

    #[tokio::test]
    async fn test_hippocampus_batch_memorize() {
        let storage = InMemoryStorage::new();
        let hippoe = Hippocampus::new(storage).unwrap();

        let mem1 = Memory::text("m1", make_embedding(&[1.0, 0.0, 0.0]), now());
        let mem2 = Memory::text("m2", make_embedding(&[0.9, 0.1, 0.0]), now());
        let mem3 = Memory::text("m3", make_embedding(&[0.8, 0.2, 0.0]), now());

        hippoe.memorize_batch(vec![mem1, mem2, mem3]).await.unwrap();

        assert_eq!(hippoe.len(), 3);

        let graph = hippoe.get_graph().await.unwrap();
        assert!(graph.edge_count() > 0);
    }

    #[tokio::test]
    async fn test_hippocampus_forget() {
        let storage = InMemoryStorage::new();
        let hippoe = Hippocampus::new(storage).unwrap();

        let mem = Memory::text("test", make_embedding(&[1.0, 0.0, 0.0]), now());
        let id = mem.id;

        hippoe.memorize(mem).await.unwrap();
        assert_eq!(hippoe.len(), 1);

        hippoe.forget(id).await.unwrap();
        assert_eq!(hippoe.len(), 0);
    }
}
