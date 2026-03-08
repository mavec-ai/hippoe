mod in_memory;

#[cfg(any(feature = "sqlite", feature = "redis", feature = "postgres"))]
mod common;

pub use in_memory::InMemoryStorage;

#[cfg(feature = "sqlite")]
mod sqlite;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteStorage;

#[cfg(feature = "redis")]
mod redis;

#[cfg(feature = "redis")]
pub use redis::RedisStorage;

#[cfg(feature = "postgres")]
mod postgres;

#[cfg(feature = "postgres")]
pub use postgres::PostgresStorage;

use crate::error::Result;
use crate::memory::Trace;
use crate::types::{Id, Link};
use async_trait::async_trait;

#[async_trait]
pub trait Storage: Send + Sync {
    async fn get(&self, id: Id) -> Result<Option<Trace>>;
    async fn put(&self, trace: Trace) -> Result<()>;
    async fn remove(&self, id: Id) -> Result<()>;
    async fn all(&self) -> Result<Vec<Trace>>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    async fn links(&self) -> Result<Vec<Link>> {
        Ok(Vec::new())
    }
}
