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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: &[f64]) -> crate::types::Embedding {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[test]
    fn test_hippocampus_default() {
        let hippoe = Hippocampus::new().unwrap();
        let config = hippoe.config();

        assert!(config.decay_rate > 0.0);
        assert!(config.spread_depth > 0);
    }

    #[tokio::test]
    async fn test_hippocampus_memorize_and_recall() {
        let hippoe = Hippocampus::new().unwrap();

        let mem1 = Trace::new(Id::new(), make_embedding(&[1.0, 0.0, 0.0]));
        let mem2 = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]));
        let mem3 = Trace::new(Id::new(), make_embedding(&[0.1, 0.9, 0.0]));

        hippoe.memorize(mem1.clone()).await.unwrap();
        hippoe.memorize(mem2.clone()).await.unwrap();
        hippoe.memorize(mem3.clone()).await.unwrap();

        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let matches = hippoe.recall(probe).await.unwrap();

        assert_eq!(matches.len(), 3);
        assert!(matches[0].score.similarity > matches[2].score.similarity);
    }

    #[tokio::test]
    async fn test_hippocampus_with_custom_storage() {
        let storage = InMemoryStorage::new();
        let hippoe = Hippocampus::builder()
            .decay_rate(0.3)
            .spread_depth(2)
            .build(storage)
            .unwrap();

        let mem = Trace::new(Id::new(), make_embedding(&[1.0, 0.0, 0.0])).emotion(0.8, 0.9);

        hippoe.memorize(mem).await.unwrap();

        let probe = make_embedding(&[0.95, 0.05, 0.0]);
        let matches = hippoe.recall(probe).await.unwrap();

        assert_eq!(matches.len(), 1);
        assert!(matches[0].score.emotion > 1.0);
    }

    #[tokio::test]
    async fn test_hippocampus_forget() {
        let hippoe = Hippocampus::new().unwrap();

        let id = Id::new();
        let mem = Trace::new(id, make_embedding(&[1.0, 0.0, 0.0]));

        hippoe.memorize(mem).await.unwrap();
        assert_eq!(hippoe.len(), 1);

        hippoe.forget(id).await.unwrap();
        assert_eq!(hippoe.len(), 0);
    }

    #[tokio::test]
    async fn test_hippocampus_get() {
        let hippoe = Hippocampus::new().unwrap();

        let id = Id::new();
        let mem = Trace::new(id, make_embedding(&[1.0, 0.0, 0.0])).emotion(0.8, 0.6);

        hippoe.memorize(mem.clone()).await.unwrap();

        let retrieved = hippoe.get(id).await.unwrap().unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.emotion.valence, 0.8);
    }

    #[tokio::test]
    async fn test_hippocampus_all() {
        let hippoe = Hippocampus::new().unwrap();

        let mem1 = Trace::new(Id::new(), make_embedding(&[1.0, 0.0, 0.0]));
        let mem2 = Trace::new(Id::new(), make_embedding(&[0.0, 1.0, 0.0]));

        hippoe.memorize(mem1).await.unwrap();
        hippoe.memorize(mem2).await.unwrap();

        let all = hippoe.all().await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_hippocampus_builder() {
        let hippoe = Hippocampus::builder()
            .decay_rate(0.3)
            .spread_depth(3)
            .min_score(0.05)
            .max_results(20)
            .boost_cap(1.5)
            .emotion_weight(0.2)
            .context_weight(0.15)
            .use_temporal_spreading(true)
            .build(InMemoryStorage::new())
            .unwrap();

        let config = hippoe.config();
        assert_eq!(config.decay_rate, 0.3);
        assert_eq!(config.spread_depth, 3);
        assert_eq!(config.min_score, 0.05);
        assert_eq!(config.max_results, 20);
        assert_eq!(config.boost_cap, 1.5);
        assert_eq!(config.emotion_weight, 0.2);
        assert_eq!(config.context_weight, 0.15);
        assert!(config.use_temporal_spreading);
    }
}
