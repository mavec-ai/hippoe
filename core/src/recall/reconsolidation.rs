use crate::config::ReconsolidationConfig;
use crate::memory::Trace;
use crate::types::{Id, Timestamp};

pub type ReconsolidationUpdates = Vec<(Id, f64, f64)>;

pub fn apply_reconsolidation(
    traces: &mut [Trace],
    activations: &[f64],
    id_to_idx: &std::collections::HashMap<Id, usize>,
    current_time: Timestamp,
    config: &ReconsolidationConfig,
) -> Vec<(Id, f64, f64)> {
    if !config.enabled {
        return Vec::new();
    }

    let mut updated = Vec::new();

    for (id, &idx) in id_to_idx {
        let activation = activations[idx];

        if activation < config.theta_low || activation > config.theta_high {
            continue;
        }

        let trace = &mut traces[idx];
        let old_access_count = trace.accesses.len();

        trace.accesses.push(current_time);

        let new_access_count = trace.accesses.len();

        updated.push((*id, old_access_count as f64, new_access_count as f64));
    }

    updated
}
