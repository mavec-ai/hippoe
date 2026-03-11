use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub min_score: f64,
    pub max_results: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            min_score: 0.01,
            max_results: 10,
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
    min_score: Option<f64>,
    max_results: Option<usize>,
}

impl ConfigBuilder {
    pub fn min_score(mut self, score: f64) -> Self {
        self.min_score = Some(score);
        self
    }

    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = Some(max);
        self
    }

    pub fn build(self) -> crate::error::Result<Config> {
        let min_score = self.min_score.unwrap_or(0.01);

        if min_score < 0.0 {
            return Err(crate::error::Error::InvalidMinScore(min_score));
        }

        Ok(Config {
            min_score,
            max_results: self.max_results.unwrap_or(10),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config::default();

        assert_eq!(config.min_score, 0.01);
        assert_eq!(config.max_results, 10);
    }

    #[test]
    fn test_config_builder_valid() {
        let config = Config::builder()
            .min_score(0.05)
            .max_results(20)
            .build()
            .unwrap();

        assert_eq!(config.min_score, 0.05);
        assert_eq!(config.max_results, 20);
    }

    #[test]
    fn test_config_builder_invalid_min_score() {
        let result = Config::builder().min_score(-0.5).build();
        assert!(result.is_err());
    }
}
