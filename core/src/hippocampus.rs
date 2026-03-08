use crate::config::Config;
use crate::error::Result;
use crate::recall::{recall, Query};

pub struct Hippocampus {
    config: Config,
}

impl Hippocampus {
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    pub fn with_config(config: Config) -> Self {
        Self { config }
    }

    pub fn builder() -> HippocampusBuilder {
        HippocampusBuilder::default()
    }

    pub fn recall(&self, query: Query) -> Result<crate::recall::RecallResult> {
        recall(query, &self.config)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl Default for Hippocampus {
    fn default() -> Self {
        Self::new()
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

    pub fn build(self) -> Result<Hippocampus> {
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
        
        Ok(Hippocampus { config: config.build()? })
    }
}
