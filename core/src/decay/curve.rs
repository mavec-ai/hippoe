use crate::types::Timestamp;

#[derive(Debug, Clone, Copy)]
pub enum Curve {
    Exponential { rate: f64 },
    PowerLaw { exponent: f64 },
    Linear { slope: f64 },
}

impl Curve {
    pub fn decay(&self, last_access: Timestamp, now: Timestamp) -> f64 {
        if now <= last_access {
            return 1.0;
        }
        let elapsed = (now - last_access) as f64 / 1000.0;

        match self {
            Curve::Exponential { rate } => (-rate * elapsed).exp(),
            Curve::PowerLaw { exponent } => 1.0 / (elapsed + 1.0).powf(*exponent),
            Curve::Linear { slope } => (1.0 - slope * elapsed).max(0.0),
        }
    }
}

impl Default for Curve {
    fn default() -> Self {
        Curve::Exponential { rate: 0.5 }
    }
}
