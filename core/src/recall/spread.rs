use std::collections::{HashMap, HashSet};

use crate::memory::Trace;
use crate::types::{Id, Link};

pub fn spread(
    memories: &[Trace],
    links: &[Link],
    initial_scores: &[f64],
    id_to_idx: &HashMap<Id, usize>,
    max_depth: usize,
) -> Vec<f64> {
    let mut scores = initial_scores.to_vec();

    let mut outgoing: HashMap<Id, Vec<(Id, f64)>> = HashMap::new();
    for link in links {
        outgoing
            .entry(link.from)
            .or_default()
            .push((link.to, link.strength));
    }

    let mut active: HashSet<Id> = memories
        .iter()
        .enumerate()
        .filter(|(i, _)| initial_scores[*i] > 0.1)
        .map(|(_, m)| m.id)
        .collect();

    for _ in 0..max_depth {
        if active.is_empty() {
            break;
        }

        let mut next_active = HashSet::new();

        for from_id in &active {
            let from_idx = match id_to_idx.get(from_id) {
                Some(&i) => i,
                None => continue,
            };

            let from_score = scores[from_idx];
            if from_score < 0.01 {
                continue;
            }

            if let Some(neighbors) = outgoing.get(from_id) {
                for (to_id, strength) in neighbors {
                    if let Some(&to_idx) = id_to_idx.get(to_id) {
                        let fan = neighbors.len() as f64;
                        let spread_amount = (from_score * strength) / fan;
                        scores[to_idx] += spread_amount * 0.5;
                        next_active.insert(*to_id);
                    }
                }
            }
        }

        active = next_active;
    }

    scores
}
