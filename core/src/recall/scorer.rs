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

pub fn similarity_batch(probe: &[f64], targets: &[&[f64]]) -> Vec<f64> {
    if probe.is_empty() {
        return vec![0.0; targets.len()];
    }

    let probe_norm: f64 = probe.iter().map(|x| x * x).sum::<f64>().sqrt();
    if probe_norm == 0.0 {
        return vec![0.0; targets.len()];
    }

    targets
        .iter()
        .map(|target| {
            if target.len() != probe.len() {
                return 0.0;
            }

            let dot: f64 = probe.iter().zip(target.iter()).map(|(p, t)| p * t).sum();
            let target_norm: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();

            if target_norm == 0.0 {
                return 0.0;
            }

            let cos = dot / (probe_norm * target_norm);
            cos.powi(3)
        })
        .collect()
}

#[allow(dead_code)]
#[inline]
pub fn probability(score: f64, temperature: f64) -> f64 {
    (score / temperature).exp()
}
