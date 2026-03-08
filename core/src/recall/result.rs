use serde::{Deserialize, Serialize};

use crate::types::Id;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scores {
    pub similarity: f64,
    pub history: f64,
    pub spread: f64,
    pub temporal: f64,
    pub boost: f64,
    pub emotion: f64,
    pub context: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    pub id: Id,
    pub score: Scores,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub matches: Vec<Match>,
    pub total_memories: usize,
}

impl RecallResult {
    pub fn matches(&self) -> &[Match] {
        &self.matches
    }

    pub fn top(&self, n: usize) -> &[Match] {
        let end = n.min(self.matches.len());
        &self.matches[..end]
    }

    pub fn first(&self) -> Option<&Match> {
        self.matches.first()
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Match> {
        self.matches.iter()
    }

    pub fn ids(&self) -> Vec<Id> {
        self.matches.iter().map(|m| m.id).collect()
    }
}
