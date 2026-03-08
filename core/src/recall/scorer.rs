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
