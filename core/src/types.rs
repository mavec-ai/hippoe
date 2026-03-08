use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type Embedding = Vec<f64>;
pub type Timestamp = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(pub Uuid);

impl Id {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Emotion {
    pub valence: f64,
    pub arousal: f64,
}

impl Default for Emotion {
    fn default() -> Self {
        Self {
            valence: 0.5,
            arousal: 0.5,
        }
    }
}

impl Emotion {
    pub fn new(valence: f64, arousal: f64) -> Self {
        Self {
            valence: valence.clamp(0.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
        }
    }

    pub fn weight(&self) -> f64 {
        (self.valence.abs() + self.arousal) / 2.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LinkKind {
    Semantic,
    Episodic,
    Causal,
    Temporal,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub from: Id,
    pub to: Id,
    pub strength: f64,
    pub kind: LinkKind,
}

impl Link {
    fn new(from: Id, to: Id, strength: f64, kind: LinkKind) -> Self {
        Self {
            from,
            to,
            strength: strength.clamp(0.0, 1.0),
            kind,
        }
    }

    pub fn semantic(from: Id, to: Id, strength: f64) -> Self {
        Self::new(from, to, strength, LinkKind::Semantic)
    }

    pub fn episodic(from: Id, to: Id, strength: f64) -> Self {
        Self::new(from, to, strength, LinkKind::Episodic)
    }

    pub fn causal(from: Id, to: Id, strength: f64) -> Self {
        Self::new(from, to, strength, LinkKind::Causal)
    }

    pub fn temporal(from: Id, to: Id, strength: f64) -> Self {
        Self::new(from, to, strength, LinkKind::Temporal)
    }
}

pub fn now() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as Timestamp
}
