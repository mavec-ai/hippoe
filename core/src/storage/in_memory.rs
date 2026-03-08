use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::Storage;
use crate::error::Result;
use crate::memory::Trace;
use crate::types::{Id, Link};
use async_trait::async_trait;

pub struct InMemoryStorage {
    traces: Arc<RwLock<HashMap<Id, Trace>>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            traces: Arc::new(RwLock::new(HashMap::new())),
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
    async fn get(&self, id: Id) -> Result<Option<Trace>> {
        let traces = self.traces.read().await;
        Ok(traces.get(&id).cloned())
    }

    async fn put(&self, trace: Trace) -> Result<()> {
        let mut traces = self.traces.write().await;
        traces.insert(trace.id, trace);
        Ok(())
    }

    async fn remove(&self, id: Id) -> Result<()> {
        let mut traces = self.traces.write().await;
        traces.remove(&id);
        Ok(())
    }

    async fn all(&self) -> Result<Vec<Trace>> {
        let traces = self.traces.read().await;
        Ok(traces.values().cloned().collect())
    }

    fn len(&self) -> usize {
        let traces = self.traces.try_read();
        match traces {
            Ok(t) => t.len(),
            Err(_) => 0,
        }
    }

    async fn links(&self) -> Result<Vec<Link>> {
        let traces = self.traces.read().await;
        let mut all_links = Vec::new();
        for trace in traces.values() {
            for (to, strength) in &trace.outgoing {
                all_links.push(Link::semantic(trace.id, *to, *strength));
            }
        }
        Ok(all_links)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: &[f64]) -> crate::types::Embedding {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[tokio::test]
    async fn test_inmemory_storage_basic_operations() {
        let storage = InMemoryStorage::new();

        assert!(storage.is_empty());
        assert_eq!(storage.len(), 0);

        let id = Id::new();
        let trace = Trace::new(id, make_embedding(&[1.0, 0.0, 0.0]));

        storage.put(trace.clone()).await.unwrap();

        assert!(!storage.is_empty());
        assert_eq!(storage.len(), 1);

        let retrieved = storage.get(id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, id);

        let all = storage.all().await.unwrap();
        assert_eq!(all.len(), 1);
    }

    #[tokio::test]
    async fn test_inmemory_storage_update() {
        let storage = InMemoryStorage::new();

        let id = Id::new();
        let trace1 = Trace::new(id, make_embedding(&[1.0, 0.0, 0.0])).accessed(1000);
        let trace2 = Trace::new(id, make_embedding(&[0.9, 0.1, 0.0])).accessed(2000);

        storage.put(trace1).await.unwrap();
        storage.put(trace2.clone()).await.unwrap();

        assert_eq!(storage.len(), 1);

        let retrieved = storage.get(id).await.unwrap().unwrap();
        assert_eq!(retrieved.last_access(), Some(2000));
    }

    #[tokio::test]
    async fn test_inmemory_storage_remove() {
        let storage = InMemoryStorage::new();

        let id1 = Id::new();
        let id2 = Id::new();

        storage
            .put(Trace::new(id1, make_embedding(&[1.0, 0.0, 0.0])))
            .await
            .unwrap();
        storage
            .put(Trace::new(id2, make_embedding(&[0.0, 1.0, 0.0])))
            .await
            .unwrap();

        assert_eq!(storage.len(), 2);

        storage.remove(id1).await.unwrap();

        assert_eq!(storage.len(), 1);
        assert!(storage.get(id1).await.unwrap().is_none());
        assert!(storage.get(id2).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_inmemory_storage_links() {
        let storage = InMemoryStorage::new();

        let id1 = Id::new();
        let id2 = Id::new();
        let id3 = Id::new();

        let trace1 = Trace::new(id1, make_embedding(&[1.0, 0.0, 0.0]))
            .link(id2, 0.8)
            .link(id3, 0.5);
        let trace2 = Trace::new(id2, make_embedding(&[0.0, 1.0, 0.0])).link(id3, 0.9);

        storage.put(trace1).await.unwrap();
        storage.put(trace2).await.unwrap();

        let links = storage.links().await.unwrap();

        assert_eq!(links.len(), 3);

        let id1_links: Vec<_> = links.iter().filter(|l| l.from == id1).collect();
        assert_eq!(id1_links.len(), 2);

        let id2_links: Vec<_> = links.iter().filter(|l| l.from == id2).collect();
        assert_eq!(id2_links.len(), 1);
    }

    #[tokio::test]
    async fn test_inmemory_storage_multiple_traces() {
        let storage = InMemoryStorage::new();

        let count = 10;
        for i in 0..count {
            let id = Id::new();
            let embedding = make_embedding(&[i as f64, 0.0, 0.0]);
            storage.put(Trace::new(id, embedding)).await.unwrap();
        }

        assert_eq!(storage.len(), count);

        let all = storage.all().await.unwrap();
        assert_eq!(all.len(), count);
    }

    #[tokio::test]
    async fn test_inmemory_storage_get_nonexistent() {
        let storage = InMemoryStorage::new();

        let result = storage.get(Id::new()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_inmemory_storage_all_empty() {
        let storage = InMemoryStorage::new();

        let all = storage.all().await.unwrap();
        assert!(all.is_empty());
    }

    #[tokio::test]
    async fn test_inmemory_storage_links_empty() {
        let storage = InMemoryStorage::new();

        let links = storage.links().await.unwrap();
        assert!(links.is_empty());
    }

    #[tokio::test]
    async fn test_inmemory_storage_remove_nonexistent() {
        let storage = InMemoryStorage::new();

        let result = storage.remove(Id::new()).await;
        assert!(result.is_ok());
        assert_eq!(storage.len(), 0);
    }
}
