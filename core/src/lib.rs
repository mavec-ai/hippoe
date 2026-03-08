pub mod config;
pub mod decay;
pub mod error;
pub mod hippocampus;
pub mod memory;
pub mod recall;
pub mod storage;
pub mod types;

pub use config::Config;
pub use decay::{Curve, boost, history_score, time_decay};
pub use hippocampus::{Hippocampus, HippocampusBuilder};
pub use memory::Trace;
pub use recall::Query;
pub use recall::scorer::{similarity, similarity_batch};
pub use storage::InMemoryStorage;
pub use storage::Storage;
pub use types::{Embedding, Emotion, Id, Link, LinkKind, Timestamp};

#[cfg(feature = "sqlite")]
pub use storage::SqliteStorage;
