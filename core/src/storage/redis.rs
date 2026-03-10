use bincode;
use redis::AsyncCommands;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    Storage,
    common::{data_to_memory, memory_to_data},
};
use crate::error::{Error, Result};
use crate::memory::{AssociationGraph, Memory};
use crate::types::Id;
use async_trait::async_trait;

const MEMORIES_PREFIX: &str = "hippoe:memory:";

pub struct RedisStorage {
    conn: Arc<RwLock<redis::aio::ConnectionManager>>,
    graph: Arc<RwLock<AssociationGraph>>,
}

impl RedisStorage {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| Error::Storage(format!("Redis client error: {}", e)))?;

        let conn = redis::aio::ConnectionManager::new(client)
            .await
            .map_err(|e| Error::Storage(format!("Redis connection error: {}", e)))?;

        Ok(Self {
            conn: Arc::new(RwLock::new(conn)),
            graph: Arc::new(RwLock::new(AssociationGraph::new())),
        })
    }

    fn key_for_memory(id: Id) -> String {
        format!("{}{}", MEMORIES_PREFIX, id)
    }
}

#[async_trait]
impl Storage for RedisStorage {
    async fn get(&self, id: Id) -> Result<Option<Memory>> {
        let mut conn = self.conn.write().await;
        let key = Self::key_for_memory(id);

        let data: Option<Vec<u8>> = conn
            .get(&key)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        match data {
            Some(bytes) => {
                let memory_data = bincode::deserialize(&bytes)
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                Ok(Some(data_to_memory(memory_data)?))
            }
            None => Ok(None),
        }
    }

    async fn put(&self, memory: Memory) -> Result<()> {
        let mut conn = self.conn.write().await;
        let key = Self::key_for_memory(memory.id);

        let memory_data = memory_to_data(&memory);
        let bytes =
            bincode::serialize(&memory_data).map_err(|e| Error::Serialization(e.to_string()))?;

        let _: () = conn
            .set(&key, bytes)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    async fn remove(&self, id: Id) -> Result<()> {
        let mut conn = self.conn.write().await;
        let key = Self::key_for_memory(id);

        let _: () = conn
            .del(&key)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    async fn all(&self) -> Result<Vec<Memory>> {
        let mut conn = self.conn.write().await;

        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(format!("{}*", MEMORIES_PREFIX))
            .query_async(&mut *conn)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.clone();
        let memories: Vec<Memory> =
            futures::future::try_join_all(keys.into_iter().map(move |key| {
                let conn = conn.clone();
                async move {
                    let mut conn = conn.write().await;
                    let data: Option<Vec<u8>> = conn
                        .get(&key)
                        .await
                        .map_err(|e| Error::Storage(e.to_string()))?;

                    match data {
                        Some(bytes) => {
                            let memory_data = bincode::deserialize(&bytes)
                                .map_err(|e| Error::Serialization(e.to_string()))?;
                            Ok(Some(data_to_memory(memory_data)?))
                        }
                        None => Ok(None),
                    }
                }
            }))
            .await?
            .into_iter()
            .flatten()
            .collect();

        Ok(memories)
    }

    fn len(&self) -> usize {
        0
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
        let memories = self.all().await?;
        Ok(memories
            .into_iter()
            .filter(|m| m.metadata.tags.iter().any(|t| t == tag))
            .collect())
    }

    async fn find_by_context(&self, context: &str) -> Result<Vec<Memory>> {
        let memories = self.all().await?;
        Ok(memories
            .into_iter()
            .filter(|m| m.metadata.context.as_ref() == Some(&context.to_string()))
            .collect())
    }

    async fn find_by_similarity(
        &self,
        embedding: &[f64],
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        let memories = self.all().await?;
        let mut scored: Vec<(f64, Memory)> = memories
            .into_iter()
            .filter_map(|m| {
                if m.embedding.len() != embedding.len() {
                    return None;
                }
                let dot: f64 = m
                    .embedding
                    .iter()
                    .zip(embedding.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let norm_a: f64 = m.embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_a > 0.0 && norm_b > 0.0 {
                    let sim = dot / (norm_a * norm_b);
                    if sim >= threshold {
                        return Some((sim, m));
                    }
                }
                None
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(_, m)| m).collect())
    }

    async fn get_associated(&self, id: Id, max_depth: usize) -> Result<Vec<Memory>> {
        let graph = self.get_graph().await?;
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((id, 0));
        visited.insert(id);

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth > 0
                && let Some(memory) = self.get(current_id).await?
            {
                result.push(memory);
            }
            if depth < max_depth {
                for neighbor in graph.neighbors(current_id, None) {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }
}
