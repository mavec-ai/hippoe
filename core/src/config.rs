use serde::{Deserialize, Serialize};

use crate::recall::TemporalConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconsolidationConfig {
    pub theta_low: f64,
    pub theta_high: f64,
    pub beta: f64,
    pub enabled: bool,
}

impl Default for ReconsolidationConfig {
    fn default() -> Self {
        Self {
            theta_low: 0.1,
            theta_high: 0.5,
            beta: 2.0,
            enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub decay_rate: f64,
    pub spread_depth: usize,
    pub min_score: f64,
    pub max_results: usize,
    pub boost_cap: f64,
    pub emotion_weight: f64,
    pub context_weight: f64,
    pub temporal: TemporalConfig,
    pub reconsolidation: ReconsolidationConfig,
    pub use_temporal_spreading: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            decay_rate: 0.5,
            spread_depth: 2,
            min_score: 0.01,
            max_results: 10,
            boost_cap: 2.0,
            emotion_weight: 0.3,
            context_weight: 0.5,
            temporal: TemporalConfig::default(),
            reconsolidation: ReconsolidationConfig::default(),
            use_temporal_spreading: true,
        }
    }
}

impl Config {
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct ConfigBuilder {
    decay_rate: Option<f64>,
    spread_depth: Option<usize>,
    min_score: Option<f64>,
    max_results: Option<usize>,
    boost_cap: Option<f64>,
    emotion_weight: Option<f64>,
    context_weight: Option<f64>,
    temporal: Option<TemporalConfig>,
    reconsolidation: Option<ReconsolidationConfig>,
    use_temporal_spreading: Option<bool>,
}

impl ConfigBuilder {
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

    pub fn temporal(mut self, config: TemporalConfig) -> Self {
        self.temporal = Some(config);
        self
    }

    pub fn reconsolidation(mut self, config: ReconsolidationConfig) -> Self {
        self.reconsolidation = Some(config);
        self
    }

    pub fn use_temporal_spreading(mut self, enabled: bool) -> Self {
        self.use_temporal_spreading = Some(enabled);
        self
    }

    pub fn build(self) -> crate::error::Result<Config> {
        let decay_rate = self.decay_rate.unwrap_or(0.5);
        let min_score = self.min_score.unwrap_or(0.01);
        let boost_cap = self.boost_cap.unwrap_or(2.0);
        let emotion_weight = self.emotion_weight.unwrap_or(0.3);
        let context_weight = self.context_weight.unwrap_or(0.5);

        if decay_rate <= 0.0 {
            return Err(crate::error::Error::InvalidDecayRate(decay_rate));
        }
        if min_score < 0.0 {
            return Err(crate::error::Error::InvalidMinScore(min_score));
        }
        if boost_cap < 1.0 {
            return Err(crate::error::Error::InvalidBoostCap(boost_cap));
        }
        if emotion_weight < 0.0 {
            return Err(crate::error::Error::InvalidEmotionWeight(emotion_weight));
        }
        if context_weight < 0.0 {
            return Err(crate::error::Error::InvalidContextWeight(context_weight));
        }

        Ok(Config {
            decay_rate,
            spread_depth: self.spread_depth.unwrap_or(2),
            min_score,
            max_results: self.max_results.unwrap_or(10),
            boost_cap,
            emotion_weight,
            context_weight,
            temporal: self.temporal.unwrap_or_default(),
            reconsolidation: self.reconsolidation.unwrap_or_default(),
            use_temporal_spreading: self.use_temporal_spreading.unwrap_or(true),
        })
    }
}
