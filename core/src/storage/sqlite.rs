use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    Storage,
    common::{trace_to_row, tuple_to_trace},
};
use crate::error::{Error, Result};
use crate::memory::Trace;
use crate::types::{Id, Link};
use async_trait::async_trait;

pub struct SqliteStorage {
    pool: Arc<RwLock<SqlitePool>>,
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
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                accesses BLOB NOT NULL,
                emotion_valence REAL NOT NULL,
                emotion_arousal REAL NOT NULL,
                wm_accessed_at INTEGER,
                outgoing BLOB NOT NULL,
                context TEXT
            )
        "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(Self {
            pool: Arc::new(RwLock::new(pool)),
        })
    }

    pub async fn new_in_memory() -> Result<Self> {
        Self::new("sqlite::memory:").await
    }
}

#[async_trait]
impl Storage for SqliteStorage {
    async fn get(&self, id: Id) -> Result<Option<Trace>> {
        let pool = self.pool.read().await;
        let row = sqlx::query_as::<_, (String, Vec<u8>, Vec<u8>, f64, f64, Option<i64>, Vec<u8>, Option<String>)>(
            "SELECT id, embedding, accesses, emotion_valence, emotion_arousal, wm_accessed_at, outgoing, context FROM traces WHERE id = ?"
        )
        .bind(id.to_string())
        .fetch_optional(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        match row {
            Some((
                id,
                embedding,
                accesses,
                emotion_valence,
                emotion_arousal,
                wm_accessed_at,
                outgoing,
                context,
            )) => Ok(Some(tuple_to_trace(
                id,
                embedding,
                accesses,
                emotion_valence,
                emotion_arousal,
                wm_accessed_at,
                outgoing,
                context,
            )?)),
            None => Ok(None),
        }
    }

    async fn put(&self, trace: Trace) -> Result<()> {
        let pool = self.pool.read().await;
        let row = trace_to_row(&trace)?;

        sqlx::query(r#"
            INSERT OR REPLACE INTO traces 
            (id, embedding, accesses, emotion_valence, emotion_arousal, wm_accessed_at, outgoing, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&row.id)
        .bind(&row.embedding)
        .bind(&row.accesses)
        .bind(row.emotion_valence)
        .bind(row.emotion_arousal)
        .bind(row.wm_accessed_at)
        .bind(&row.outgoing)
        .bind(&row.context)
        .execute(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    async fn remove(&self, id: Id) -> Result<()> {
        let pool = self.pool.read().await;
        sqlx::query("DELETE FROM traces WHERE id = ?")
            .bind(id.to_string())
            .execute(&*pool)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    async fn all(&self) -> Result<Vec<Trace>> {
        let pool = self.pool.read().await;
        let rows = sqlx::query_as::<_, (String, Vec<u8>, Vec<u8>, f64, f64, Option<i64>, Vec<u8>, Option<String>)>(
            "SELECT id, embedding, accesses, emotion_valence, emotion_arousal, wm_accessed_at, outgoing, context FROM traces"
        )
        .fetch_all(&*pool)
        .await
        .map_err(|e| Error::Storage(e.to_string()))?;

        rows.into_iter()
            .map(
                |(
                    id,
                    embedding,
                    accesses,
                    emotion_valence,
                    emotion_arousal,
                    wm_accessed_at,
                    outgoing,
                    context,
                )| {
                    tuple_to_trace(
                        id,
                        embedding,
                        accesses,
                        emotion_valence,
                        emotion_arousal,
                        wm_accessed_at,
                        outgoing,
                        context,
                    )
                },
            )
            .collect()
    }

    fn len(&self) -> usize {
        0
    }

    async fn links(&self) -> Result<Vec<Link>> {
        let traces = self.all().await?;
        let mut all_links = Vec::new();
        for trace in traces {
            for (to, strength) in &trace.outgoing {
                all_links.push(Link::semantic(trace.id, *to, *strength));
            }
        }
        Ok(all_links)
    }
}
