use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub decay_rate: f64,
    pub min_score: f64,
    pub max_results: usize,
    pub emotion_weight: f64,
    pub context_weight: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            decay_rate: 0.5,
            min_score: 0.01,
            max_results: 10,
            emotion_weight: 0.3,
            context_weight: 0.5,
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
    min_score: Option<f64>,
    max_results: Option<usize>,
    emotion_weight: Option<f64>,
    context_weight: Option<f64>,
}

impl ConfigBuilder {
    pub fn decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = Some(rate);
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

    pub fn emotion_weight(mut self, weight: f64) -> Self {
        self.emotion_weight = Some(weight);
        self
    }

    pub fn context_weight(mut self, weight: f64) -> Self {
        self.context_weight = Some(weight);
        self
    }

    pub fn build(self) -> crate::error::Result<Config> {
        let decay_rate = self.decay_rate.unwrap_or(0.5);
        let min_score = self.min_score.unwrap_or(0.01);
        let emotion_weight = self.emotion_weight.unwrap_or(0.3);
        let context_weight = self.context_weight.unwrap_or(0.5);

        if decay_rate <= 0.0 {
            return Err(crate::error::Error::InvalidDecayRate(decay_rate));
        }
        if min_score < 0.0 {
            return Err(crate::error::Error::InvalidMinScore(min_score));
        }
        if emotion_weight < 0.0 {
            return Err(crate::error::Error::InvalidEmotionWeight(emotion_weight));
        }
        if context_weight < 0.0 {
            return Err(crate::error::Error::InvalidContextWeight(context_weight));
        }

        Ok(Config {
            decay_rate,
            min_score,
            max_results: self.max_results.unwrap_or(10),
            emotion_weight,
            context_weight,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config::default();

        assert_eq!(config.decay_rate, 0.5);
        assert_eq!(config.min_score, 0.01);
        assert_eq!(config.max_results, 10);
        assert_eq!(config.emotion_weight, 0.3);
        assert_eq!(config.context_weight, 0.5);
    }

    #[test]
    fn test_config_builder_valid() {
        let config = Config::builder()
            .decay_rate(0.8)
            .min_score(0.05)
            .max_results(20)
            .build()
            .unwrap();

        assert_eq!(config.decay_rate, 0.8);
        assert_eq!(config.min_score, 0.05);
        assert_eq!(config.max_results, 20);
    }

    #[test]
    fn test_config_builder_invalid_decay_rate() {
        let result = Config::builder().decay_rate(-0.1).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_builder_invalid_min_score() {
        let result = Config::builder().min_score(-0.5).build();
        assert!(result.is_err());
    }
}
