//! Core type definitions for hippoe-core.
//!
//! This module provides the fundamental types used throughout the memory system:
//!
//! - [`Id`]: Unique identifier for memories
//! - [`Emotion`]: Emotional dimensions based on the Circumplex Model of Affect
//! - [`Link`] and [`LinkKind`]: Association types between memories
//! - [`Embedding`]: Vector representation type alias
//! - [`Timestamp`]: Time representation
//!
//! # Circumplex Model of Affect
//!
//! The [`Emotion`] struct implements Russell's Circumplex Model (1980), which
//! represents emotional states as points in a 2D space defined by:
//!
//! - **Valence**: Pleasantness dimension (negative to positive)
//! - **Arousal**: Activation dimension (calm to excited)
//!
//! Reference: Russell, J. A. (1980). A circumplex model of affect.
//! *Journal of Personality and Social Psychology*, 39(6), 1161-1178.
//! DOI: 10.1037/h0077714

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

impl std::str::FromStr for Id {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Id(Uuid::parse_str(s)?))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_generation() {
        let id1 = Id::new();
        let id2 = Id::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_emotion_weight() {
        let emotion = Emotion::new(0.5, 0.8);
        assert!(emotion.weight() > 0.0);

        let neutral = Emotion::default();
        assert!(neutral.weight() < emotion.weight());
    }

    #[test]
    fn test_link_kind() {
        let id1 = Id::new();
        let id2 = Id::new();

        let link_semantic = Link::semantic(id1, id2, 0.5);
        let link_episodic = Link::episodic(id1, id2, 0.6);
        let link_causal = Link::causal(id1, id2, 0.7);
        let link_temporal = Link::temporal(id1, id2, 0.3);

        assert_eq!(link_semantic.kind, LinkKind::Semantic);
        assert_eq!(link_episodic.kind, LinkKind::Episodic);
        assert_eq!(link_causal.kind, LinkKind::Causal);
        assert_eq!(link_temporal.kind, LinkKind::Temporal);
    }
}
