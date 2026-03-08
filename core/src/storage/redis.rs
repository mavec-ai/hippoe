use bincode;
use redis::AsyncCommands;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    Storage,
    common::{TraceData, data_to_trace, trace_to_data},
};
use crate::error::{Error, Result};
use crate::memory::Trace;
use crate::types::{Id, Link};
use async_trait::async_trait;

const TRACES_PREFIX: &str = "hippoe:trace:";
const LINKS_SET: &str = "hippoe:links";

pub struct RedisStorage {
    conn: Arc<RwLock<redis::aio::ConnectionManager>>,
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
        })
    }

    fn key_for_trace(id: Id) -> String {
        format!("{}{}", TRACES_PREFIX, id)
    }
}

#[async_trait]
impl Storage for RedisStorage {
    async fn get(&self, id: Id) -> Result<Option<Trace>> {
        let mut conn = self.conn.write().await;
        let key = Self::key_for_trace(id);

        let data: Option<Vec<u8>> = conn
            .get(&key)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        match data {
            Some(bytes) => {
                let trace_data: TraceData = bincode::deserialize(&bytes)
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                Ok(Some(data_to_trace(trace_data)?))
            }
            None => Ok(None),
        }
    }

    async fn put(&self, trace: Trace) -> Result<()> {
        let mut conn = self.conn.write().await;
        let key = Self::key_for_trace(trace.id);

        let trace_data = trace_to_data(&trace);
        let bytes =
            bincode::serialize(&trace_data).map_err(|e| Error::Serialization(e.to_string()))?;

        let _: () = conn
            .set(&key, bytes)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        for (to, strength) in &trace.outgoing {
            let link_key = format!("{}:{}", trace.id, to);
            let _: () = conn
                .hset(LINKS_SET, &link_key, *strength)
                .await
                .map_err(|e| Error::Storage(e.to_string()))?;
        }

        Ok(())
    }

    async fn remove(&self, id: Id) -> Result<()> {
        let mut conn = self.conn.write().await;
        let key = Self::key_for_trace(id);

        let _: () = conn
            .del(&key)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    async fn all(&self) -> Result<Vec<Trace>> {
        let mut conn = self.conn.write().await;

        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(format!("{}*", TRACES_PREFIX))
            .query_async(&mut *conn)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.clone();
        let traces: Vec<Trace> = futures::future::try_join_all(keys.into_iter().map(move |key| {
            let conn = conn.clone();
            async move {
                let mut conn = conn.write().await;
                let data: Option<Vec<u8>> = conn
                    .get(&key)
                    .await
                    .map_err(|e| Error::Storage(e.to_string()))?;

                match data {
                    Some(bytes) => {
                        let trace_data: TraceData = bincode::deserialize(&bytes)
                            .map_err(|e| Error::Serialization(e.to_string()))?;
                        Ok(Some(data_to_trace(trace_data)?))
                    }
                    None => Ok(None),
                }
            }
        }))
        .await?
        .into_iter()
        .flatten()
        .collect();

        Ok(traces)
    }

    fn len(&self) -> usize {
        0
    }

    async fn links(&self) -> Result<Vec<Link>> {
        let mut conn = self.conn.write().await;

        let link_entries: Vec<(String, f64)> = conn
            .hgetall(LINKS_SET)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        let mut links = Vec::new();
        for (key, strength) in link_entries {
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() == 2
                && let (Ok(from), Ok(to)) = (parts[0].parse::<Id>(), parts[1].parse::<Id>())
            {
                links.push(Link::semantic(from, to, strength));
            }
        }

        Ok(links)
    }
}
