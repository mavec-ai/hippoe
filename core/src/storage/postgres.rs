use sqlx::FromRow;
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    Storage,
    common::{memory_to_row, row_to_memory},
};
use crate::error::{Error, Result};
use crate::memory::{AssociationGraph, Memory};
use crate::types::Id;
use async_trait::async_trait;

#[derive(Debug, FromRow)]
struct MemoryRowPg {
    id: String,
    content_text: Option<String>,
    content_structured: Option<String>,
    content_raw: Option<Vec<u8>>,
    embedding: Vec<u8>,
    created_at: i64,
    updated_at: i64,
    last_accessed_at: i64,
    access_count: i64,
    access_timestamps: Vec<u8>,
    emotion_valence: f64,
    emotion_arousal: f64,
    decay_rate: f64,
    importance: f64,
    context: Option<String>,
    tags: Vec<u8>,
    lability: f64,
    consolidation_threshold: f64,
    consolidation_state: String,
    last_consolidation_at: i64,
    associations: Vec<u8>,
    temporal_links: Vec<u8>,
}

pub struct PostgresStorage {
    pool: Arc<RwLock<PgPool>>,
    graph: Arc<RwLock<AssociationGraph>>,
}

impl PostgresStorage {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
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
                content_raw BYTEA,
                embedding BYTEA NOT NULL,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL,
                last_accessed_at BIGINT NOT NULL,
                access_count BIGINT NOT NULL,
                access_timestamps BYTEA NOT NULL,
                emotion_valence DOUBLE PRECISION NOT NULL,
                emotion_arousal DOUBLE PRECISION NOT NULL,
                decay_rate DOUBLE PRECISION NOT NULL,
                importance DOUBLE PRECISION NOT NULL,
                context TEXT,
                tags BYTEA NOT NULL,
                lability DOUBLE PRECISION NOT NULL,
                consolidation_threshold DOUBLE PRECISION NOT NULL,
                consolidation_state TEXT NOT NULL DEFAULT 'fresh',
                last_consolidation_at BIGINT NOT NULL DEFAULT 0,
                associations BYTEA NOT NULL,
                temporal_links BYTEA NOT NULL
            )
        "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(Self {
            pool: Arc::new(RwLock::new(pool)),
            graph: Arc::new(RwLock::new(AssociationGraph::new())),
        })
    }
}

fn pg_row_to_memory_row(row: MemoryRowPg) -> super::common::MemoryRow {
    super::common::MemoryRow {
        id: row.id,
        content_text: row.content_text,
        content_structured: row.content_structured,
        content_raw: row.content_raw,
        embedding: row.embedding,
        created_at: row.created_at as u64,
        updated_at: row.updated_at as u64,
        last_accessed_at: row.last_accessed_at as u64,
        access_count: row.access_count as u64,
        access_timestamps: row.access_timestamps,
        emotion_valence: row.emotion_valence,
        emotion_arousal: row.emotion_arousal,
        decay_rate: row.decay_rate,
        importance: row.importance,
        context: row.context,
        tags: row.tags,
        lability: row.lability,
        consolidation_threshold: row.consolidation_threshold,
        consolidation_state: row.consolidation_state,
        last_consolidation_at: row.last_consolidation_at as u64,
        associations: row.associations,
        temporal_links: row.temporal_links,
    }
}

#[async_trait]
impl Storage for PostgresStorage {
    async fn get(&self, id: Id) -> Result<Option<Memory>> {
        let pool = self.pool.read().await;
        let row: Option<MemoryRowPg> = sqlx::query_as(
            "SELECT id, content_text, content_structured, content_raw, embedding, created_at, updated_at, last_accessed_at, access_count, access_timestamps, emotion_valence, emotion_arousal, decay_rate, importance, context, tags, lability, consolidation_threshold, consolidation_state, last_consolidation_at, associations, temporal_links FROM memories WHERE id = $1"
        )
        .bind(id.to_string())
        .fetch_optional(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        match row {
            Some(r) => Ok(Some(row_to_memory(pg_row_to_memory_row(r))?)),
            None => Ok(None),
        }
    }

    async fn put(&self, memory: Memory) -> Result<()> {
        let pool = self.pool.read().await;
        let row = memory_to_row(&memory)?;

        sqlx::query(r#"
            INSERT INTO memories 
            (id, content_text, content_structured, content_raw, embedding, created_at, updated_at, last_accessed_at, access_count, access_timestamps, emotion_valence, emotion_arousal, decay_rate, importance, context, tags, lability, consolidation_threshold, consolidation_state, last_consolidation_at, associations, temporal_links)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
            ON CONFLICT (id) DO UPDATE SET
                content_text = EXCLUDED.content_text,
                content_structured = EXCLUDED.content_structured,
                content_raw = EXCLUDED.content_raw,
                embedding = EXCLUDED.embedding,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at,
                last_accessed_at = EXCLUDED.last_accessed_at,
                access_count = EXCLUDED.access_count,
                access_timestamps = EXCLUDED.access_timestamps,
                emotion_valence = EXCLUDED.emotion_valence,
                emotion_arousal = EXCLUDED.emotion_arousal,
                decay_rate = EXCLUDED.decay_rate,
                importance = EXCLUDED.importance,
                context = EXCLUDED.context,
                tags = EXCLUDED.tags,
                lability = EXCLUDED.lability,
                consolidation_threshold = EXCLUDED.consolidation_threshold,
                consolidation_state = EXCLUDED.consolidation_state,
                last_consolidation_at = EXCLUDED.last_consolidation_at,
                associations = EXCLUDED.associations,
                temporal_links = EXCLUDED.temporal_links
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
        .bind(&row.consolidation_state)
        .bind(row.last_consolidation_at as i64)
        .bind(&row.associations)
        .bind(&row.temporal_links)
        .execute(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    async fn remove(&self, id: Id) -> Result<()> {
        let pool = self.pool.read().await;
        sqlx::query("DELETE FROM memories WHERE id = $1")
            .bind(id.to_string())
            .execute(&*pool)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    async fn all(&self) -> Result<Vec<Memory>> {
        let pool = self.pool.read().await;
        let rows: Vec<MemoryRowPg> = sqlx::query_as(
            "SELECT id, content_text, content_structured, content_raw, embedding, created_at, updated_at, last_accessed_at, access_count, access_timestamps, emotion_valence, emotion_arousal, decay_rate, importance, context, tags, lability, consolidation_threshold, consolidation_state, last_consolidation_at, associations, temporal_links FROM memories"
        )
        .fetch_all(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        rows.into_iter()
            .map(|r| row_to_memory(pg_row_to_memory_row(r)))
            .collect()
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
