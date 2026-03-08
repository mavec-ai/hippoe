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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_decay() {
        let now: crate::types::Timestamp = 100000;

        let decay_recent = time_decay(90000, now, 0.5);
        let decay_old = time_decay(50000, now, 0.5);

        assert!(decay_recent > decay_old);
        assert!(decay_recent <= 1.0);
        assert!(decay_old >= 0.0);

        let decay_future = time_decay(110000, now, 0.5);
        assert_eq!(decay_future, 1.0);
    }

    #[test]
    fn test_boost() {
        let now: crate::types::Timestamp = 100000;

        let boost_recent = boost(90000, now, 2.0);
        let boost_old = boost(50000, now, 2.0);

        assert!(boost_recent > boost_old);
        assert!(boost_recent <= 2.0);

        let boost_future = boost(110000, now, 2.0);
        assert_eq!(boost_future, 2.0);
    }
}
