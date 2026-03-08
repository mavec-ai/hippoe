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
