mod curve;
mod history;

pub use curve::Curve;
pub use history::history_score;

#[inline]
pub fn time_decay(last_access: u64, now: u64, rate: f64) -> f64 {
    if now <= last_access {
        return 1.0;
    }
    let elapsed = (now - last_access) as f64 / 1000.0;
    (-rate * elapsed).exp()
}

#[inline]
pub fn boost(accessed_at: u64, now: u64, cap: f64) -> f64 {
    if now <= accessed_at {
        return cap;
    }
    let age_ms = now - accessed_at;
    let decay = (age_ms as f64 / 30_000.0).ln().max(0.0);
    (cap - decay * 0.1).max(1.0)
}
