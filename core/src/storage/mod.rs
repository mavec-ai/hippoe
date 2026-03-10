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
use crate::memory::{AssociationGraph, Memory};
use crate::types::Id;
use async_trait::async_trait;

#[async_trait]
pub trait Storage: Send + Sync {
    async fn get(&self, id: Id) -> Result<Option<Memory>>;
    async fn put(&self, memory: Memory) -> Result<()>;
    async fn remove(&self, id: Id) -> Result<()>;
    async fn all(&self) -> Result<Vec<Memory>>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    async fn get_graph(&self) -> Result<AssociationGraph>;
    async fn update_graph<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut AssociationGraph) + Send;
    async fn find_by_tag(&self, tag: &str) -> Result<Vec<Memory>>;
    async fn find_by_context(&self, context: &str) -> Result<Vec<Memory>>;
    async fn find_by_similarity(
        &self,
        embedding: &[f64],
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<Memory>>;
    async fn get_associated(&self, id: Id, max_depth: usize) -> Result<Vec<Memory>>;
}
