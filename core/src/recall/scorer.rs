//! Similarity and activation functions for memory retrieval.
//!
//! This module implements core similarity and activation algorithms:
//! - MINERVA 2 cubed similarity (Hintzman, 1986)
//! - Cosine similarity with `\[0,1\]` clamping
//! - Multiplicative activation combination
//! - Surprise computation for reconsolidation triggers
//!
//! # MINERVA 2 Activation
//!
//! Formula: `similarity = max(0, cos³(a, b))` where:
//! - `cos(a, b)` = cosine similarity between vectors
//! - Cubing amplifies differences and emphasizes strong matches
//! - Negative values clamped to 0
//!
//! # Batch Processing
//!
//! Use `similarity_batch()` and `cosine_similarity_batch()` for retrieval hot path.
//! Pre-computed probe norm improves performance by avoiding redundant calculations.
//!
//! # Multiplicative Combination
//!
//! Combines multiple activation sources: `activation = product of factors`
//! Ensures all factors must be present for high activation.
//!
//! # References
//!
//! - Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. DOI:10.1037/0033-295X.93.4.411

#[inline]
pub fn similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let (dot, norm_a, norm_b) = a
        .iter()
        .zip(b.iter())
        .fold((0.0, 0.0, 0.0), |(d, na, nb), (&x, &y)| {
            (d + x * y, na + x * x, nb + y * y)
        });

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    let cos = dot / denom;
    cos.powi(3).max(0.0)
}

#[inline]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let (dot, norm_a, norm_b) = a
        .iter()
        .zip(b.iter())
        .fold((0.0, 0.0, 0.0), |(d, na, nb), (&x, &y)| {
            (d + x * y, na + x * x, nb + y * y)
        });

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    (dot / denom).max(0.0)
}

pub fn similarity_batch(probe: &[f64], targets: &[&[f64]]) -> Vec<f64> {
    if probe.is_empty() {
        return vec![0.0; targets.len()];
    }

    let probe_norm_sq: f64 = probe.iter().map(|x| x * x).sum();
    if probe_norm_sq == 0.0 {
        return vec![0.0; targets.len()];
    }

    targets
        .iter()
        .map(|target| {
            if target.len() != probe.len() {
                return 0.0;
            }

            let (dot, target_norm_sq) = probe
                .iter()
                .zip(target.iter())
                .fold((0.0, 0.0), |(d, tn), (&p, &t)| {
                    (p.mul_add(t, d), t.mul_add(t, tn))
                });

            if target_norm_sq == 0.0 {
                return 0.0;
            }

            let cos = dot / (probe_norm_sq * target_norm_sq).sqrt();
            cos.powi(3)
        })
        .collect()
}

pub fn cosine_similarity_batch(probe: &[f64], targets: &[&[f64]]) -> Vec<f64> {
    if probe.is_empty() {
        return vec![0.0; targets.len()];
    }

    let probe_norm_sq: f64 = probe.iter().map(|x| x * x).sum();
    if probe_norm_sq == 0.0 {
        return vec![0.0; targets.len()];
    }

    targets
        .iter()
        .map(|target| {
            if target.len() != probe.len() {
                return 0.0;
            }

            let (dot, target_norm_sq) = probe
                .iter()
                .zip(target.iter())
                .fold((0.0, 0.0), |(d, tn), (&p, &t)| {
                    (p.mul_add(t, d), t.mul_add(t, tn))
                });

            if target_norm_sq == 0.0 {
                return 0.0;
            }

            (dot / (probe_norm_sq * target_norm_sq).sqrt()).max(0.0)
        })
        .collect()
}

#[inline]
pub fn retrieval_probability(total_activation: f64, threshold: f64, noise: f64) -> f64 {
    if noise <= 0.0 {
        return if total_activation >= threshold {
            1.0
        } else {
            0.0
        };
    }
    1.0 / (1.0 + (-(total_activation - threshold) / noise).exp())
}

#[inline]
pub fn retrieval_probability_batch(activations: &[f64], threshold: f64, noise: f64) -> Vec<f64> {
    if noise <= 0.0 {
        return activations
            .iter()
            .map(|&a| if a >= threshold { 1.0 } else { 0.0 })
            .collect();
    }
    let inv_noise = 1.0 / noise;
    activations
        .iter()
        .map(|&a| 1.0 / (1.0 + (-(a - threshold) * inv_noise).exp()))
        .collect()
}

#[inline]
pub fn combine_activations_multiplicative(
    probe_activation: f64,
    base_level: f64,
    spreading: f64,
    emotional_weight: f64,
) -> f64 {
    let emotional_multiplier = 1.0 + (emotional_weight - 0.5);

    let effective_base = if base_level.is_finite() {
        base_level
    } else {
        -10.0
    };

    let recency_boost = ((effective_base + 10.0) / 10.0).clamp(0.0, 1.0);

    let modulated_probe = probe_activation * emotional_multiplier;

    modulated_probe * (1.0 + recency_boost) + spreading
}

#[inline]
pub fn compute_surprise(
    expected: &[f64],
    actual: &[f64],
    memory_age_days: f64,
    memory_strength: f64,
    base_threshold: f64,
) -> f64 {
    let sim = cosine_similarity(expected, actual);
    let prediction_error = 1.0 - sim;

    let age_factor = memory_age_days * 0.01;
    let strength_factor = memory_strength * 0.2;
    let dynamic_threshold = base_threshold + age_factor + strength_factor;

    (prediction_error / dynamic_threshold).clamp(0.0, 1.0)
}

#[inline]
pub fn triggers_lability(surprise: f64, threshold: f64) -> bool {
    surprise > threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: &[f64]) -> crate::types::Embedding {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[test]
    fn test_similarity_identical() {
        let a = make_embedding(&[1.0, 0.0, 0.0]);
        let b = make_embedding(&[1.0, 0.0, 0.0]);

        let sim = similarity(&a, &b);
        assert!(sim > 0.99);
    }

    #[test]
    fn test_similarity_orthogonal() {
        let a = make_embedding(&[1.0, 0.0, 0.0]);
        let b = make_embedding(&[0.0, 1.0, 0.0]);

        let sim = similarity(&a, &b);
        assert!(sim < 0.01);
    }

    #[test]
    fn test_similarity_opposite() {
        let a = make_embedding(&[1.0, 0.0, 0.0]);
        let b = make_embedding(&[-1.0, 0.0, 0.0]);

        let sim = similarity(&a, &b);
        assert!(sim < 0.01);
    }

    #[test]
    fn test_similarity_batch() {
        let probe = make_embedding(&[1.0, 0.0, 0.0]);
        let t1 = make_embedding(&[1.0, 0.0, 0.0]);
        let t2 = make_embedding(&[0.0, 1.0, 0.0]);
        let t3 = make_embedding(&[0.707, 0.707, 0.0]);

        let targets: Vec<&[f64]> = vec![&t1, &t2, &t3];

        let sims = similarity_batch(&probe, &targets);

        assert_eq!(sims.len(), 3);
        assert!(sims[0] > 0.99);
        assert!(sims[1] < 0.01);
        assert!(sims[2] > 0.3 && sims[2] < 0.5);
    }
}
