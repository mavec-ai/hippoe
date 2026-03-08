use crate::config::Config;
use crate::error::Result;
use crate::memory::Trace;
use crate::recall::RecallResult;
use crate::recall::{Query, recall};
use crate::storage::{InMemoryStorage, Storage};
use crate::types::{Embedding, Id, now};

pub struct Hippocampus<S: Storage> {
    config: Config,
    storage: S,
}

impl Hippocampus<InMemoryStorage> {
    pub fn new() -> Result<Self> {
        HippocampusBuilder::default().build(InMemoryStorage::new())
    }

    pub fn builder() -> HippocampusBuilder {
        HippocampusBuilder::default()
    }
}

impl<S: Storage> Hippocampus<S> {
    pub fn recall_with_query(&self, query: Query) -> Result<RecallResult> {
        recall(query, &self.config)
    }

    pub async fn memorize(&self, trace: Trace) -> Result<()> {
        self.storage.put(trace).await
    }

    pub async fn recall(&self, probe: Embedding) -> Result<Vec<crate::recall::Match>> {
        let traces = self.storage.all().await?;
        let links = self.storage.links().await?;

        let query = Query::new(probe).memories(&traces).links(&links).now(now());

        let result = recall(query, &self.config)?;

        self.apply_reconsolidation(&result).await?;

        Ok(result.matches().to_vec())
    }

    pub async fn forget(&self, id: Id) -> Result<()> {
        self.storage.remove(id).await
    }

    pub async fn get(&self, id: Id) -> Result<Option<Trace>> {
        self.storage.get(id).await
    }

    pub async fn all(&self) -> Result<Vec<Trace>> {
        self.storage.all().await
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    async fn apply_reconsolidation(&self, result: &RecallResult) -> Result<()> {
        for m in result.matches() {
            if let Some(mut trace) = self.storage.get(m.id).await? {
                trace.accesses.push(now());
                self.storage.put(trace).await?;
            }
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct HippocampusBuilder {
    decay_rate: Option<f64>,
    spread_depth: Option<usize>,
    min_score: Option<f64>,
    max_results: Option<usize>,
    boost_cap: Option<f64>,
    emotion_weight: Option<f64>,
    context_weight: Option<f64>,
    use_temporal_spreading: Option<bool>,
}

impl HippocampusBuilder {
    pub fn decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = Some(rate);
        self
    }

    pub fn spread_depth(mut self, depth: usize) -> Self {
        self.spread_depth = Some(depth);
        self
    }

    pub fn min_score(mut self, score: f64) -> Self {
        self.min_score = Some(score);
        self
    }

    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = Some(max);
        self
    }

    pub fn boost_cap(mut self, cap: f64) -> Self {
        self.boost_cap = Some(cap);
        self
    }

    pub fn emotion_weight(mut self, weight: f64) -> Self {
        self.emotion_weight = Some(weight);
        self
    }

    pub fn context_weight(mut self, weight: f64) -> Self {
        self.context_weight = Some(weight);
        self
    }

    pub fn use_temporal_spreading(mut self, enabled: bool) -> Self {
        self.use_temporal_spreading = Some(enabled);
        self
    }

    pub fn build<S: Storage>(self, storage: S) -> Result<Hippocampus<S>> {
        let mut config = Config::builder();

        if let Some(v) = self.decay_rate {
            config = config.decay_rate(v);
        }
        if let Some(v) = self.spread_depth {
            config = config.spread_depth(v);
        }
        if let Some(v) = self.min_score {
            config = config.min_score(v);
        }
        if let Some(v) = self.max_results {
            config = config.max_results(v);
        }
        if let Some(v) = self.boost_cap {
            config = config.boost_cap(v);
        }
        if let Some(v) = self.emotion_weight {
            config = config.emotion_weight(v);
        }
        if let Some(v) = self.context_weight {
            config = config.context_weight(v);
        }
        if let Some(v) = self.use_temporal_spreading {
            config = config.use_temporal_spreading(v);
        }

        Ok(Hippocampus {
            config: config.build()?,
            storage,
        })
    }
}
