mod query;
mod reconsolidation;
mod result;
pub mod scorer;
mod spread;
pub mod temporal;

pub use query::Query;
pub use reconsolidation::{ReconsolidationUpdates, apply_reconsolidation};
pub use result::{Match, RecallResult, Scores};
pub use scorer::{similarity, similarity_batch};
pub use spread::spread;
pub use temporal::{TemporalConfig, TemporalLink, create_episode_links, spread_temporal};

use crate::config::Config;
use crate::decay::{boost, history_score};
use crate::error;
use crate::types::{Id, now};

struct RecallIntermediate {
    matches: Vec<Match>,
    all_activations: Vec<f64>,
    total_memories: usize,
}

fn recall_internal(query: Query, config: &Config) -> error::Result<RecallIntermediate> {
    if query.probe.is_empty() {
        return Err(error::Error::EmptyProbe);
    }
    if query.memories.is_empty() {
        return Err(error::Error::NoMemories);
    }

    let dim = query.probe.len();
    for m in &query.memories {
        if m.embedding.len() != dim {
            return Err(error::Error::DimensionMismatch {
                expected: dim,
                actual: m.embedding.len(),
            });
        }
    }

    let current_time = query.now.unwrap_or_else(now);

    let embeddings: Vec<&[f64]> = query
        .memories
        .iter()
        .map(|m| m.embedding.as_slice())
        .collect();
    let similarities = similarity_batch(&query.probe, &embeddings);

    let id_to_idx: std::collections::HashMap<Id, usize> = query
        .memories
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id, i))
        .collect();

    let mut spread_scores = vec![0.0; query.memories.len()];
    if !query.links.is_empty() && config.spread_depth > 0 {
        spread_scores = spread(
            &query.memories,
            &query.links,
            &similarities,
            &id_to_idx,
            config.spread_depth,
        );
    }

    let mut temporal_scores = vec![0.0; query.memories.len()];
    if config.use_temporal_spreading && !query.temporal_links.is_empty() {
        temporal_scores = spread_temporal(
            &query.memories,
            &query.temporal_links,
            &similarities,
            &id_to_idx,
            current_time,
            &config.temporal,
        );
    }

    let matches: Vec<Match> = query
        .memories
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let history = history_score(&m.accesses, current_time, config.decay_rate);

            let wm_boost = m
                .wm_accessed_at
                .map(|t| boost(t, current_time, config.boost_cap))
                .unwrap_or(1.0);

            let emotion_factor = 1.0 + (m.emotion.weight() * config.emotion_weight);

            let context_factor = match (&query.context, &m.context) {
                (Some(qc), Some(mc)) if qc == mc => 1.0 + config.context_weight,
                _ => 1.0,
            };

            let total =
                (similarities[i] + history + spread_scores[i] + temporal_scores[i] + wm_boost)
                    * emotion_factor
                    * context_factor;

            Match {
                id: m.id,
                score: Scores {
                    similarity: similarities[i],
                    history,
                    spread: spread_scores[i],
                    temporal: temporal_scores[i],
                    boost: wm_boost,
                    emotion: emotion_factor,
                    context: context_factor,
                    total,
                },
                probability: 0.0,
            }
        })
        .collect();

    let all_activations: Vec<f64> = matches.iter().map(|m| m.score.total).collect();

    Ok(RecallIntermediate {
        matches,
        all_activations,
        total_memories: query.memories.len(),
    })
}

pub fn recall(query: Query, config: &Config) -> error::Result<RecallResult> {
    let mut intermediate = recall_internal(query, config)?;

    let n = config.max_results.min(intermediate.matches.len());
    if n < intermediate.matches.len() {
        intermediate
            .matches
            .select_nth_unstable_by(n, |a, b| b.score.total.partial_cmp(&a.score.total).unwrap());
        intermediate.matches.truncate(n);
    } else {
        intermediate
            .matches
            .sort_by(|a, b| b.score.total.partial_cmp(&a.score.total).unwrap());
    }

    intermediate
        .matches
        .retain(|m| m.score.total >= config.min_score);

    let exp_scores: Vec<f64> = intermediate
        .matches
        .iter()
        .map(|m| m.score.total.exp())
        .collect();
    let prob_total: f64 = exp_scores.iter().sum();
    for (m, exp_score) in intermediate.matches.iter_mut().zip(exp_scores.iter()) {
        m.probability = exp_score / prob_total;
    }

    Ok(RecallResult {
        matches: intermediate.matches,
        total_memories: intermediate.total_memories,
    })
}

pub fn recall_with_reconsolidation(
    mut query: Query,
    config: &Config,
) -> error::Result<(RecallResult, ReconsolidationUpdates)> {
    let current_time = query.now.unwrap_or_else(now);
    let intermediate = recall_internal(query.clone(), config)?;

    let id_to_idx: std::collections::HashMap<Id, usize> = query
        .memories
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id, i))
        .collect();

    let reconsolidation_updates = apply_reconsolidation(
        &mut query.memories,
        &intermediate.all_activations,
        &id_to_idx,
        current_time,
        &config.reconsolidation,
    );

    let mut intermediate = intermediate;
    let n = config.max_results.min(intermediate.matches.len());
    if n < intermediate.matches.len() {
        intermediate
            .matches
            .select_nth_unstable_by(n, |a, b| b.score.total.partial_cmp(&a.score.total).unwrap());
        intermediate.matches.truncate(n);
    } else {
        intermediate
            .matches
            .sort_by(|a, b| b.score.total.partial_cmp(&a.score.total).unwrap());
    }

    intermediate
        .matches
        .retain(|m| m.score.total >= config.min_score);

    let exp_scores: Vec<f64> = intermediate
        .matches
        .iter()
        .map(|m| m.score.total.exp())
        .collect();
    let prob_total: f64 = exp_scores.iter().sum();
    for (m, exp_score) in intermediate.matches.iter_mut().zip(exp_scores.iter()) {
        m.probability = exp_score / prob_total;
    }

    Ok((
        RecallResult {
            matches: intermediate.matches,
            total_memories: intermediate.total_memories,
        },
        reconsolidation_updates,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Trace;
    use crate::types::Link;

    fn make_embedding(values: &[f64]) -> crate::types::Embedding {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[test]
    fn test_recall_basic() {
        let config = Config::default();
        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let mem1 = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0])).accessed(1000);
        let mem2 = Trace::new(Id::new(), make_embedding(&[0.1, 0.9, 0.0])).accessed(1000);

        let query = Query::new(probe)
            .add_memory(mem1)
            .add_memory(mem2)
            .now(2000);

        let result = recall(query, &config).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.matches()[0].score.similarity > result.matches()[1].score.similarity);
    }

    #[test]
    fn test_recall_with_spreading() {
        let config = Config::default();
        let id1 = Id::new();
        let id2 = Id::new();
        let id3 = Id::new();

        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let mem1 = Trace::new(id1, make_embedding(&[0.9, 0.1, 0.0]));
        let mem2 = Trace::new(id2, make_embedding(&[0.1, 0.9, 0.0]));
        let mem3 = Trace::new(id3, make_embedding(&[0.2, 0.8, 0.0]));

        let link = Link::semantic(id1, id2, 0.8);

        let query = Query::new(probe)
            .add_memory(mem1)
            .add_memory(mem2)
            .add_memory(mem3)
            .add_link(link)
            .now(1000);

        let result = recall(query, &config).unwrap();

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_recall_empty_probe_error() {
        let config = Config::default();
        let query = Query::new(Vec::new());

        let result = recall(query, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_recall_no_memories_error() {
        let config = Config::default();
        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let query = Query::new(probe);

        let result = recall(query, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_recall_dimension_mismatch_error() {
        let config = Config::default();
        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let mem = Trace::new(Id::new(), make_embedding(&[1.0, 0.0, 0.0, 0.0]));

        let query = Query::new(probe).add_memory(mem);

        let result = recall(query, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_recall_with_emotion() {
        let config = Config {
            emotion_weight: 0.5,
            ..Config::default()
        };

        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let mem_neutral = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]));
        let mem_emotional = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0])).emotion(0.9, 0.9);

        let query = Query::new(probe.clone())
            .add_memory(mem_neutral.clone())
            .add_memory(mem_emotional.clone())
            .now(1000);

        let result = recall(query, &config).unwrap();

        let emotional_idx = result
            .matches()
            .iter()
            .position(|m| m.id == mem_emotional.id);

        if let Some(idx) = emotional_idx {
            assert!(result.matches()[idx].score.emotion > 1.0);
        }
    }

    #[test]
    fn test_recall_with_working_memory() {
        let config = Config {
            boost_cap: 2.0,
            ..Config::default()
        };

        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let mem_wm = Trace::new(Id::new(), make_embedding(&[0.7, 0.3, 0.0])).wm_accessed(900);
        let mem_no_wm = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]));

        let query = Query::new(probe)
            .add_memory(mem_wm)
            .add_memory(mem_no_wm)
            .now(1000);

        let result = recall(query, &config).unwrap();

        assert!(result.matches().iter().any(|m| m.score.boost > 1.0));
    }

    #[test]
    fn test_recall_probability_sums() {
        let config = Config::default();
        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let mem1 = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]));
        let mem2 = Trace::new(Id::new(), make_embedding(&[0.8, 0.2, 0.0]));
        let mem3 = Trace::new(Id::new(), make_embedding(&[0.7, 0.3, 0.0]));

        let query = Query::new(probe)
            .add_memory(mem1)
            .add_memory(mem2)
            .add_memory(mem3)
            .now(1000);

        let result = recall(query, &config).unwrap();

        let prob_sum: f64 = result.matches().iter().map(|m| m.probability).sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_min_score_filtering() {
        let config = Config {
            min_score: 0.5,
            ..Config::default()
        };

        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let mem_high = Trace::new(Id::new(), make_embedding(&[0.99, 0.01, 0.0]));
        let mem_low = Trace::new(Id::new(), make_embedding(&[0.0, 0.99, 0.0]));

        let query = Query::new(probe)
            .add_memory(mem_high)
            .add_memory(mem_low)
            .now(1000);

        let result = recall(query, &config).unwrap();

        for m in result.matches() {
            assert!(m.score.total >= 0.5);
        }
    }

    #[test]
    fn test_max_results_limit() {
        let config = Config {
            max_results: 2,
            ..Config::default()
        };

        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let memories: Vec<Trace> = (0..10)
            .map(|i| {
                Trace::new(
                    Id::new(),
                    make_embedding(&[0.9 - i as f64 * 0.05, 0.1, 0.0]),
                )
            })
            .collect();

        let mut query = Query::new(probe);
        for m in memories {
            query = query.add_memory(m);
        }
        query = query.now(1000);

        let result = recall(query, &config).unwrap();

        assert!(result.len() <= 2);
    }
}
