use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::memory::Trace;
use crate::types::{Id, Timestamp};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalLink {
    pub from: Id,
    pub to: Id,
    pub strength: f64,
    pub distance: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    pub forward_strength: f64,
    pub backward_strength: f64,
    pub distance_decay: f64,
    pub max_distance: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            forward_strength: 1.0,
            backward_strength: 0.7,
            distance_decay: 0.3,
            max_distance: 5,
        }
    }
}

#[inline]
pub fn compute_temporal_strength(
    distance: usize,
    is_forward: bool,
    config: &TemporalConfig,
) -> f64 {
    let base = if is_forward {
        config.forward_strength
    } else {
        config.backward_strength
    };
    base * (-config.distance_decay * distance as f64).exp()
}

pub fn create_episode_links(
    memories: &[Trace],
    config: &TemporalConfig,
) -> Vec<TemporalLink> {
    if memories.len() < 2 {
        return Vec::new();
    }

    let mut sorted: Vec<(usize, &Trace)> = memories
        .iter()
        .enumerate()
        .collect();

    sorted.sort_by(|a, b| {
        let time_a = a.1.last_access().unwrap_or(0);
        let time_b = b.1.last_access().unwrap_or(0);
        time_a.cmp(&time_b)
    });

    let mut links = Vec::new();
    let n = sorted.len();

    for (i, &(_, trace_i)) in sorted.iter().enumerate() {
        let window_start = i + 1;
        let window_end = (i + config.max_distance + 1).min(n);
        for (window_idx, &(_, trace_j)) in sorted[window_start..window_end].iter().enumerate() {
            let distance = window_idx + 1;

            links.push(TemporalLink {
                from: trace_i.id,
                to: trace_j.id,
                strength: compute_temporal_strength(distance, true, config),
                distance,
            });

            links.push(TemporalLink {
                from: trace_j.id,
                to: trace_i.id,
                strength: compute_temporal_strength(distance, false, config),
                distance,
            });
        }
    }

    links
}

pub fn spread_temporal(
    memories: &[Trace],
    temporal_links: &[TemporalLink],
    initial_scores: &[f64],
    id_to_idx: &HashMap<Id, usize>,
    current_time: Timestamp,
    config: &TemporalConfig,
) -> Vec<f64> {
    let mut scores = initial_scores.to_vec();

    let mut links_by_source: HashMap<Id, Vec<&TemporalLink>> = HashMap::new();
    for link in temporal_links {
        links_by_source
            .entry(link.from)
            .or_default()
            .push(link);
    }

    for (idx, trace) in memories.iter().enumerate() {
        let base_score = initial_scores[idx];
        if base_score < 0.01 {
            continue;
        }

        if let Some(links) = links_by_source.get(&trace.id) {
            for link in links {
                if let Some(&target_idx) = id_to_idx.get(&link.to) {
                    let time_factor = trace
                        .last_access()
                        .map(|t| {
                            let elapsed = current_time.saturating_sub(t) as f64 / 1000.0;
                            (-config.distance_decay * elapsed / 3600.0).exp()
                        })
                        .unwrap_or(1.0);

                    let spread_amount = base_score * link.strength * time_factor;
                    scores[target_idx] += spread_amount;
                }
            }
        }
    }

    scores
}

#[inline]
pub fn temporal_distance(a: &Trace, b: &Trace) -> Option<usize> {
    let time_a = a.last_access()?;
    let time_b = b.last_access()?;
    
    let diff_ms = (time_a as i64 - time_b as i64).unsigned_abs();
    let diff_secs = diff_ms / 1000;
    
    match diff_secs {
        0..=60 => Some(1),
        61..=300 => Some(2),
        301..=900 => Some(3),
        901..=3600 => Some(4),
        _ => None,
    }
}
