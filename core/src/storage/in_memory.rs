use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::Storage;
use crate::error::Result;
use crate::memory::{AssociationGraph, Memory};
use crate::types::Id;
use async_trait::async_trait;

pub struct InMemoryStorage {
    memories: Arc<RwLock<HashMap<Id, Memory>>>,
    graph: Arc<RwLock<AssociationGraph>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            memories: Arc::new(RwLock::new(HashMap::new())),
            graph: Arc::new(RwLock::new(AssociationGraph::new())),
        }
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Storage for InMemoryStorage {
    async fn get(&self, id: Id) -> Result<Option<Memory>> {
        let memories = self.memories.read().await;
        Ok(memories.get(&id).cloned())
    }

    async fn put(&self, memory: Memory) -> Result<()> {
        let mut memories = self.memories.write().await;
        memories.insert(memory.id, memory);
        Ok(())
    }

    async fn remove(&self, id: Id) -> Result<()> {
        let mut memories = self.memories.write().await;
        let mut graph = self.graph.write().await;

        memories.remove(&id);
        graph.remove_node(id);

        Ok(())
    }

    async fn all(&self) -> Result<Vec<Memory>> {
        let memories = self.memories.read().await;
        Ok(memories.values().cloned().collect())
    }

    fn len(&self) -> usize {
        let memories = self.memories.try_read();
        match memories {
            Ok(m) => m.len(),
            Err(_) => 0,
        }
    }

    async fn get_graph(&self) -> Result<AssociationGraph> {
        let graph = self.graph.read().await;
        Ok(graph.clone())
    }

    async fn update_graph<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut AssociationGraph) + Send,
    {
        let mut graph = self.graph.write().await;
        f(&mut graph);
        Ok(())
    }

    async fn find_by_tag(&self, tag: &str) -> Result<Vec<Memory>> {
        let memories = self.memories.read().await;
        Ok(memories
            .values()
            .filter(|m| m.metadata.tags.contains(&tag.to_string()))
            .cloned()
            .collect())
    }

    async fn find_by_context(&self, context: &str) -> Result<Vec<Memory>> {
        let memories = self.memories.read().await;
        Ok(memories
            .values()
            .filter(|m| {
                m.metadata
                    .context
                    .as_ref()
                    .map(|c| c == context)
                    .unwrap_or(false)
            })
            .cloned()
            .collect())
    }

    async fn find_by_similarity(
        &self,
        embedding: &[f64],
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        let memories = self.memories.read().await;
        let mut scored: Vec<(f64, &Memory)> = memories
            .values()
            .filter_map(|m| {
                let sim = compute_cosine_similarity(embedding, &m.embedding);
                if sim >= threshold {
                    Some((sim, m))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(limit);

        Ok(scored.into_iter().map(|(_, m)| m.clone()).collect())
    }

    async fn get_associated(&self, id: Id, max_depth: usize) -> Result<Vec<Memory>> {
        let graph = self.graph.read().await;
        let memories = self.memories.read().await;

        let mut activations = graph.spreading_activation(id, max_depth, 0.5);
        
        for edge in graph.get_edges_to(id) {
            let entry = activations.entry(edge.from).or_insert(0.0);
            *entry = entry.max(edge.strength);
        }

        let mut associated: Vec<Memory> = activations
            .into_iter()
            .filter_map(|(mem_id, _)| memories.get(&mem_id).cloned())
            .collect();

        associated.retain(|m| m.id != id);

        Ok(associated)
    }
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
    use crate::memory::{AssociationEdge, MemoryBuilder};
    use crate::types::{now, LinkKind};

    fn make_embedding(values: &[f64]) -> Vec<f64> {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[tokio::test]
    async fn test_memory_storage_basic_operations() {
        let storage = InMemoryStorage::new();

        assert!(storage.is_empty());
        assert_eq!(storage.len(), 0);

        let mem = Memory::text("test", make_embedding(&[1.0, 0.0, 0.0]), now());

        storage.put(mem.clone()).await.unwrap();

        assert!(!storage.is_empty());
        assert_eq!(storage.len(), 1);

        let retrieved = storage.get(mem.id).await.unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_memory_storage_graph_operations() {
        let storage = InMemoryStorage::new();

        let mem1 = Memory::text("memory 1", make_embedding(&[1.0, 0.0, 0.0]), now());
        let mem2 = Memory::text("memory 2", make_embedding(&[0.9, 0.1, 0.0]), now());

        storage.put(mem1.clone()).await.unwrap();
        storage.put(mem2.clone()).await.unwrap();

        storage
            .update_graph(|g| {
                g.add_edge(AssociationEdge::new(
                    mem1.id,
                    mem2.id,
                    0.8,
                    LinkKind::Semantic,
                    now(),
                ));
            })
            .await
            .unwrap();

        let graph = storage.get_graph().await.unwrap();
        assert!(graph.has_edge(mem1.id, mem2.id));
    }

    #[tokio::test]
    async fn test_memory_storage_find_by_tag() {
        let storage = InMemoryStorage::new();
        let current_time = now();

        let mem1 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("important note")
            .tag("important")
            .build();

        let mem2 = MemoryBuilder::new(make_embedding(&[0.0, 1.0, 0.0]), current_time)
            .text("regular note")
            .tag("normal")
            .build();

        storage.put(mem1).await.unwrap();
        storage.put(mem2).await.unwrap();

        let important = storage.find_by_tag("important").await.unwrap();
        assert_eq!(important.len(), 1);
        assert_eq!(important[0].content.text, Some("important note".to_string()));
    }

    #[tokio::test]
    async fn test_memory_storage_find_by_context() {
        let storage = InMemoryStorage::new();
        let current_time = now();

        let mem1 = MemoryBuilder::new(make_embedding(&[1.0, 0.0, 0.0]), current_time)
            .text("work task")
            .context("work")
            .build();

        let mem2 = MemoryBuilder::new(make_embedding(&[0.0, 1.0, 0.0]), current_time)
            .text("personal task")
            .context("personal")
            .build();

        storage.put(mem1).await.unwrap();
        storage.put(mem2).await.unwrap();

        let work_memories = storage.find_by_context("work").await.unwrap();
        assert_eq!(work_memories.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_storage_find_by_similarity() {
        let storage = InMemoryStorage::new();

        let mem1 = Memory::text("m1", make_embedding(&[1.0, 0.0, 0.0]), now());
        let mem2 = Memory::text("m2", make_embedding(&[0.95, 0.1, 0.0]), now());
        let mem3 = Memory::text("m3", make_embedding(&[0.1, 0.9, 0.0]), now());

        storage.put(mem1).await.unwrap();
        storage.put(mem2).await.unwrap();
        storage.put(mem3).await.unwrap();

        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let similar = storage.find_by_similarity(&probe, 0.9, 10).await.unwrap();

        assert_eq!(similar.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_storage_get_associated() {
        let storage = InMemoryStorage::new();

        let mem1 = Memory::text("m1", make_embedding(&[1.0, 0.0, 0.0]), now());
        let mem2 = Memory::text("m2", make_embedding(&[0.9, 0.1, 0.0]), now());
        let mem3 = Memory::text("m3", make_embedding(&[0.1, 0.9, 0.0]), now());

        storage.put(mem1.clone()).await.unwrap();
        storage.put(mem2.clone()).await.unwrap();
        storage.put(mem3.clone()).await.unwrap();

        storage
            .update_graph(|g| {
                g.add_edge(AssociationEdge::new(
                    mem1.id,
                    mem2.id,
                    0.9,
                    LinkKind::Semantic,
                    now(),
                ));
                g.add_edge(AssociationEdge::new(
                    mem1.id,
                    mem3.id,
                    0.5,
                    LinkKind::Episodic,
                    now(),
                ));
            })
            .await
            .unwrap();

        let associated = storage.get_associated(mem1.id, 2).await.unwrap();
        assert_eq!(associated.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_storage_remove_cleans_graph() {
        let storage = InMemoryStorage::new();

        let mem1 = Memory::text("m1", make_embedding(&[1.0, 0.0, 0.0]), now());
        let mem2 = Memory::text("m2", make_embedding(&[0.9, 0.1, 0.0]), now());

        storage.put(mem1.clone()).await.unwrap();
        storage.put(mem2.clone()).await.unwrap();

        storage
            .update_graph(|g| {
                g.add_node(mem1.id);
                g.add_node(mem2.id);
            })
            .await
            .unwrap();

        storage.remove(mem1.id).await.unwrap();

        let graph = storage.get_graph().await.unwrap();
        assert!(!graph.has_node(mem1.id));
        assert!(graph.has_node(mem2.id));
    }
}
