use crate::error::Result;
use crate::memory::Memory;
use crate::recall::{RetrievalContext, RetrievalMatch, RetrievalStrategy};
use crate::types::{Embedding, Id, now};

#[derive(Clone)]
pub struct MemoryQuery {
    probe: Option<Embedding>,
    max_results: usize,
    min_similarity: f64,
    tags: Vec<String>,
    contexts: Vec<String>,
    min_emotion: Option<f64>,
    max_emotion: Option<f64>,
    min_importance: Option<f64>,
    exclude_ids: Vec<Id>,
    include_associations_of: Option<Id>,
    association_depth: usize,
}

impl MemoryQuery {
    pub fn new() -> Self {
        Self {
            probe: None,
            max_results: 10,
            min_similarity: 0.0,
            tags: Vec::new(),
            contexts: Vec::new(),
            min_emotion: None,
            max_emotion: None,
            min_importance: None,
            exclude_ids: Vec::new(),
            include_associations_of: None,
            association_depth: 2,
        }
    }

    pub fn similar_to(mut self, embedding: Embedding) -> Self {
        self.probe = Some(embedding);
        self
    }

    pub fn max_results(mut self, n: usize) -> Self {
        self.max_results = n.max(1);
        self
    }

    pub fn min_similarity(mut self, threshold: f64) -> Self {
        self.min_similarity = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for tag in tags {
            self.tags.push(tag.into());
        }
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.contexts.push(context.into());
        self
    }

    pub fn with_contexts(mut self, contexts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for ctx in contexts {
            self.contexts.push(ctx.into());
        }
        self
    }

    pub fn with_emotion_range(mut self, min: f64, max: f64) -> Self {
        self.min_emotion = Some(min.clamp(0.0, 1.0));
        self.max_emotion = Some(max.clamp(0.0, 1.0));
        self
    }

    pub fn with_min_importance(mut self, importance: f64) -> Self {
        self.min_importance = Some(importance.clamp(0.0, 1.0));
        self
    }

    pub fn exclude(mut self, id: Id) -> Self {
        self.exclude_ids.push(id);
        self
    }

    pub fn exclude_many(mut self, ids: impl IntoIterator<Item = Id>) -> Self {
        self.exclude_ids.extend(ids);
        self
    }

    pub fn include_associations(mut self, of_id: Id, depth: usize) -> Self {
        self.include_associations_of = Some(of_id);
        self.association_depth = depth;
        self
    }

    pub fn probe(&self) -> Option<&Embedding> {
        self.probe.as_ref()
    }

    pub fn max_results_value(&self) -> usize {
        self.max_results
    }

    pub fn min_similarity_value(&self) -> f64 {
        self.min_similarity
    }

    pub fn matches_filters(&self, memory: &Memory) -> bool {
        if !self.tags.is_empty() {
            let has_tag = self.tags.iter().any(|t| memory.metadata.tags.contains(t));
            if !has_tag {
                return false;
            }
        }

        if !self.contexts.is_empty() {
            let has_context = memory
                .metadata
                .context
                .as_ref()
                .map(|c| self.contexts.contains(c))
                .unwrap_or(false);
            if !has_context {
                return false;
            }
        }

        if let (Some(min), Some(max)) = (self.min_emotion, self.max_emotion) {
            let emotion = memory.metadata.emotional_weight.weight();
            if emotion < min || emotion > max {
                return false;
            }
        }

        if let Some(min_imp) = self.min_importance
            && memory.metadata.importance < min_imp
        {
            return false;
        }

        if self.exclude_ids.contains(&memory.id) {
            return false;
        }

        true
    }

    pub fn apply_to_memories(&self, memories: Vec<Memory>) -> Vec<Memory> {
        memories
            .into_iter()
            .filter(|m| self.matches_filters(m))
            .collect()
    }

    pub fn into_context(self) -> RetrievalContext {
        let mut ctx = RetrievalContext::new(self.probe.unwrap_or_default(), now())
            .with_max_results(self.max_results);

        if self.min_similarity > 0.0 {
            ctx = ctx.with_min_threshold(self.min_similarity);
        }

        ctx
    }
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MemoryQueryBuilder<'a, S: crate::storage::Storage> {
    hippocampus: &'a crate::Hippocampus<S>,
    query: MemoryQuery,
    strategy: Option<Box<dyn RetrievalStrategy>>,
}

impl<'a, S: crate::storage::Storage> MemoryQueryBuilder<'a, S> {
    pub(crate) fn new(hippocampus: &'a crate::Hippocampus<S>) -> Self {
        Self {
            hippocampus,
            query: MemoryQuery::new(),
            strategy: None,
        }
    }

    pub fn similar_to(mut self, embedding: Embedding) -> Self {
        self.query = self.query.similar_to(embedding);
        self
    }

    pub fn max_results(mut self, n: usize) -> Self {
        self.query = self.query.max_results(n);
        self
    }

    pub fn min_similarity(mut self, threshold: f64) -> Self {
        self.query = self.query.min_similarity(threshold);
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.query = self.query.with_tag(tag);
        self
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.query = self.query.with_tags(tags);
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.query = self.query.with_context(context);
        self
    }

    pub fn with_contexts(mut self, contexts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.query = self.query.with_contexts(contexts);
        self
    }

    pub fn with_emotion_range(mut self, min: f64, max: f64) -> Self {
        self.query = self.query.with_emotion_range(min, max);
        self
    }

    pub fn with_min_importance(mut self, importance: f64) -> Self {
        self.query = self.query.with_min_importance(importance);
        self
    }

    pub fn exclude(mut self, id: Id) -> Self {
        self.query = self.query.exclude(id);
        self
    }

    pub fn exclude_many(mut self, ids: impl IntoIterator<Item = Id>) -> Self {
        self.query = self.query.exclude_many(ids);
        self
    }

    pub fn include_associations(mut self, of_id: Id, depth: usize) -> Self {
        self.query = self.query.include_associations(of_id, depth);
        self
    }

    pub fn with_strategy(mut self, strategy: Box<dyn RetrievalStrategy>) -> Self {
        self.strategy = Some(strategy);
        self
    }

    pub async fn execute(self) -> Result<Vec<Memory>> {
        let all_memories = self.hippocampus.all().await?;
        let filtered = self.query.apply_to_memories(all_memories);

        if self.query.probe().is_some() {
            let graph = self.hippocampus.get_graph().await?;

            let context = self.query.clone().into_context();

            let matches = if let Some(strategy) = self.strategy {
                strategy.retrieve(&filtered, &graph, &context).await
            } else {
                self.hippocampus
                    .retrieve_with_default_strategy(&filtered, &graph, &context)
                    .await
            };

            let ids: std::collections::HashSet<Id> = matches.iter().map(|m| m.memory_id).collect();

            let mut results: Vec<Memory> = filtered
                .into_iter()
                .filter(|m| ids.contains(&m.id))
                .collect();

            if let Some(assoc_id) = self.query.include_associations_of {
                let associated = self
                    .hippocampus
                    .recall_associated(assoc_id, self.query.association_depth)
                    .await?;

                for mem in associated {
                    if !results.iter().any(|r| r.id == mem.id) {
                        results.push(mem);
                    }
                }
            }

            results.truncate(self.query.max_results_value());
            Ok(results)
        } else {
            let mut results = filtered;
            results.truncate(self.query.max_results_value());
            Ok(results)
        }
    }

    pub async fn execute_with_scores(self) -> Result<Vec<RetrievalMatch>> {
        let all_memories = self.hippocampus.all().await?;
        let filtered = self.query.apply_to_memories(all_memories);

        let graph = self.hippocampus.get_graph().await?;
        let context = self.query.into_context();

        let matches = if let Some(strategy) = self.strategy {
            strategy.retrieve(&filtered, &graph, &context).await
        } else {
            self.hippocampus
                .retrieve_with_default_strategy(&filtered, &graph, &context)
                .await
        };

        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_query_builder_filters_tags() {
        let now = now();
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);

        let mem_with_tag = MemoryBuilder::new(embedding.clone(), now)
            .text("important memory")
            .tag("important")
            .build();

        let mem_without_tag = MemoryBuilder::new(embedding.clone(), now)
            .text("normal memory")
            .build();

        let query = MemoryQuery::new().with_tag("important");

        assert!(query.matches_filters(&mem_with_tag));
        assert!(!query.matches_filters(&mem_without_tag));
    }

    #[test]
    fn test_query_builder_filters_context() {
        let now = now();
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);

        let mem_with_context = MemoryBuilder::new(embedding.clone(), now)
            .text("work memory")
            .context("work")
            .build();

        let mem_other_context = MemoryBuilder::new(embedding.clone(), now)
            .text("home memory")
            .context("home")
            .build();

        let query = MemoryQuery::new().with_context("work");

        assert!(query.matches_filters(&mem_with_context));
        assert!(!query.matches_filters(&mem_other_context));
    }

    #[test]
    fn test_query_builder_filters_importance() {
        let now = now();
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);

        let high_importance = MemoryBuilder::new(embedding.clone(), now)
            .text("important")
            .importance(0.9)
            .build();

        let low_importance = MemoryBuilder::new(embedding.clone(), now)
            .text("not important")
            .importance(0.3)
            .build();

        let query = MemoryQuery::new().with_min_importance(0.7);

        assert!(query.matches_filters(&high_importance));
        assert!(!query.matches_filters(&low_importance));
    }

    #[test]
    fn test_query_builder_filters_emotion() {
        let now = now();
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);

        let emotional_mem = MemoryBuilder::new(embedding.clone(), now)
            .text("emotional")
            .emotion(0.8, 0.9)
            .build();

        let neutral_mem = MemoryBuilder::new(embedding.clone(), now)
            .text("neutral")
            .emotion(0.2, 0.5)
            .build();

        let query = MemoryQuery::new().with_emotion_range(0.7, 1.0);

        assert!(query.matches_filters(&emotional_mem));
        assert!(!query.matches_filters(&neutral_mem));
    }

    #[test]
    fn test_query_builder_exclude_ids() {
        let now = now();
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);

        let mem1 = MemoryBuilder::new(embedding.clone(), now)
            .text("memory 1")
            .build();

        let mem2 = MemoryBuilder::new(embedding.clone(), now)
            .text("memory 2")
            .build();

        let query = MemoryQuery::new().exclude(mem1.id);

        assert!(!query.matches_filters(&mem1));
        assert!(query.matches_filters(&mem2));
    }

    #[test]
    fn test_query_builder_multiple_filters() {
        let now = now();
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);

        let matching_mem = MemoryBuilder::new(embedding.clone(), now)
            .text("matching")
            .tag("important")
            .context("work")
            .importance(0.9)
            .build();

        let wrong_tag = MemoryBuilder::new(embedding.clone(), now)
            .text("wrong tag")
            .tag("personal")
            .context("work")
            .importance(0.9)
            .build();

        let wrong_context = MemoryBuilder::new(embedding.clone(), now)
            .text("wrong context")
            .tag("important")
            .context("home")
            .importance(0.9)
            .build();

        let low_importance = MemoryBuilder::new(embedding.clone(), now)
            .text("low importance")
            .tag("important")
            .context("work")
            .importance(0.3)
            .build();

        let query = MemoryQuery::new()
            .with_tag("important")
            .with_context("work")
            .with_min_importance(0.7);

        assert!(query.matches_filters(&matching_mem));
        assert!(!query.matches_filters(&wrong_tag));
        assert!(!query.matches_filters(&wrong_context));
        assert!(!query.matches_filters(&low_importance));
    }

    #[test]
    fn test_query_apply_to_memories() {
        let now = now();
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);

        let memories: Vec<Memory> = (0..5)
            .map(|i| {
                MemoryBuilder::new(embedding.clone(), now)
                    .text(format!("memory {}", i))
                    .tag(if i % 2 == 0 { "even" } else { "odd" })
                    .build()
            })
            .collect();

        let query = MemoryQuery::new().with_tag("even");
        let filtered = query.apply_to_memories(memories);

        assert_eq!(filtered.len(), 3);
        for mem in &filtered {
            assert!(mem.metadata.tags.contains(&"even".to_string()));
        }
    }
}
