use crate::types::Timestamp;

const RECENT_THRESHOLD_MS: u64 = 86_400_000; // 24 jam

#[inline]
pub fn history_score(accesses: &[Timestamp], now: Timestamp, rate: f64) -> f64 {
    if accesses.is_empty() {
        return 0.0;
    }

    let sum: f64 = accesses
        .iter()
        .map(|&t| {
            let elapsed_ms = now.saturating_sub(t).max(1);
            let elapsed_s = elapsed_ms as f64 / 1000.0;

            if elapsed_ms < RECENT_THRESHOLD_MS {
                (-rate * elapsed_s).exp()
            } else {
                elapsed_s.powf(-rate)
            }
        })
        .sum();

    sum.ln().max(-10.0).exp()
}
