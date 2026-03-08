use sqlx::postgres::{PgPool, PgPoolOptions};
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

pub struct PostgresStorage {
    pool: Arc<RwLock<PgPool>>,
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
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                embedding BYTEA NOT NULL,
                accesses BYTEA NOT NULL,
                emotion_valence DOUBLE PRECISION NOT NULL,
                emotion_arousal DOUBLE PRECISION NOT NULL,
                wm_accessed_at BIGINT,
                outgoing BYTEA NOT NULL,
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
}

#[async_trait]
impl Storage for PostgresStorage {
    async fn get(&self, id: Id) -> Result<Option<Trace>> {
        let pool = self.pool.read().await;
        let row = sqlx::query_as::<_, (String, Vec<u8>, Vec<u8>, f64, f64, Option<i64>, Vec<u8>, Option<String>)>(
            "SELECT id, embedding, accesses, emotion_valence, emotion_arousal, wm_accessed_at, outgoing, context FROM traces WHERE id = $1"
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
            INSERT INTO traces 
            (id, embedding, accesses, emotion_valence, emotion_arousal, wm_accessed_at, outgoing, context)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                accesses = EXCLUDED.accesses,
                emotion_valence = EXCLUDED.emotion_valence,
                emotion_arousal = EXCLUDED.emotion_arousal,
                wm_accessed_at = EXCLUDED.wm_accessed_at,
                outgoing = EXCLUDED.outgoing,
                context = EXCLUDED.context
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
        sqlx::query("DELETE FROM traces WHERE id = $1")
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
