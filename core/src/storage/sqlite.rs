use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    Storage,
    common::{memory_to_row, row_to_memory, MemoryRow},
};
use crate::error::{Error, Result};
use crate::memory::{AssociationGraph, Memory};
use crate::types::Id;
use async_trait::async_trait;

pub struct SqliteStorage {
    pool: Arc<RwLock<SqlitePool>>,
    graph: Arc<RwLock<AssociationGraph>>,
    len: Arc<RwLock<usize>>,
}

impl SqliteStorage {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content_text TEXT,
                content_structured TEXT,
                content_raw BLOB,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_accessed_at INTEGER NOT NULL,
                access_count INTEGER NOT NULL,
                access_timestamps BLOB NOT NULL,
                emotion_valence REAL NOT NULL,
                emotion_arousal REAL NOT NULL,
                decay_rate REAL NOT NULL,
                importance REAL NOT NULL,
                context TEXT,
                tags BLOB NOT NULL,
                lability REAL NOT NULL,
                consolidation_threshold REAL NOT NULL,
                associations BLOB NOT NULL
            )
        "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(Self {
            pool: Arc::new(RwLock::new(pool)),
            graph: Arc::new(RwLock::new(AssociationGraph::new())),
            len: Arc::new(RwLock::new(0)),
        })
    }

    pub async fn new_in_memory() -> Result<Self> {
        Self::new("sqlite::memory:").await
    }
}

#[async_trait]
impl Storage for SqliteStorage {
    async fn get(&self, id: Id) -> Result<Option<Memory>> {
        let pool = self.pool.read().await;
        let row = sqlx::query_as::<_, MemoryRow>(
            "SELECT id, content_text, content_structured, content_raw, embedding, created_at, updated_at, last_accessed_at, access_count, access_timestamps, emotion_valence, emotion_arousal, decay_rate, importance, context, tags, lability, consolidation_threshold, associations FROM memories WHERE id = ?"
        )
        .bind(id.to_string())
        .fetch_optional(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        match row {
            Some(r) => Ok(Some(row_to_memory(r)?)),
            None => Ok(None),
        }
    }

    async fn put(&self, memory: Memory) -> Result<()> {
        let pool = self.pool.read().await;
        let row = memory_to_row(&memory)?;

        sqlx::query(r#"
            INSERT OR REPLACE INTO memories 
            (id, content_text, content_structured, content_raw, embedding, created_at, updated_at, last_accessed_at, access_count, access_timestamps, emotion_valence, emotion_arousal, decay_rate, importance, context, tags, lability, consolidation_threshold, associations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&row.id)
        .bind(&row.content_text)
        .bind(&row.content_structured)
        .bind(&row.content_raw)
        .bind(&row.embedding)
        .bind(row.created_at as i64)
        .bind(row.updated_at as i64)
        .bind(row.last_accessed_at as i64)
        .bind(row.access_count as i64)
        .bind(&row.access_timestamps)
        .bind(row.emotion_valence)
        .bind(row.emotion_arousal)
        .bind(row.decay_rate)
        .bind(row.importance)
        .bind(&row.context)
        .bind(&row.tags)
        .bind(row.lability)
        .bind(row.consolidation_threshold)
        .bind(&row.associations)
        .execute(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        let mut len = self.len.write().await;
        let exists: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM memories WHERE id = ?)")
            .bind(&row.id)
            .fetch_one(&*pool)
            .await
            .unwrap_or(false);
        if !exists {
            *len += 1;
        }

        Ok(())
    }

    async fn remove(&self, id: Id) -> Result<()> {
        let pool = self.pool.read().await;
        sqlx::query("DELETE FROM memories WHERE id = ?")
            .bind(id.to_string())
            .execute(&*pool)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;
        
        let mut len = self.len.write().await;
        if *len > 0 {
            *len -= 1;
        }
        
        Ok(())
    }

    async fn all(&self) -> Result<Vec<Memory>> {
        let pool = self.pool.read().await;
        let rows = sqlx::query_as::<_, MemoryRow>(
            "SELECT id, content_text, content_structured, content_raw, embedding, created_at, updated_at, last_accessed_at, access_count, access_timestamps, emotion_valence, emotion_arousal, decay_rate, importance, context, tags, lability, consolidation_threshold, associations FROM memories"
        )
        .fetch_all(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        rows.into_iter()
            .map(row_to_memory)
            .collect()
    }

    fn len(&self) -> usize {
        self.len.try_read().map(|g| *g).unwrap_or(0)
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
                let dot: f64 = m.embedding.iter().zip(embedding.iter()).map(|(a, b)| a * b).sum();
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
