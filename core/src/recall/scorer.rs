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
    cos.powi(3)
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
