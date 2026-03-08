use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{Embedding, Emotion, Id, Link, Timestamp};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub id: Id,
    pub embedding: Embedding,
    pub accesses: Vec<Timestamp>,
    pub emotion: Emotion,
    pub wm_accessed_at: Option<Timestamp>,
    pub outgoing: HashMap<Id, f64>,
    pub context: Option<String>,
}

impl Trace {
    pub fn new(id: Id, embedding: Embedding) -> Self {
        Self {
            id,
            embedding,
            accesses: Vec::new(),
            emotion: Emotion::default(),
            wm_accessed_at: None,
            outgoing: HashMap::new(),
            context: None,
        }
    }

    pub fn accessed(mut self, at: Timestamp) -> Self {
        self.accesses.push(at);
        self
    }

    pub fn emotion(mut self, valence: f64, arousal: f64) -> Self {
        self.emotion = Emotion::new(valence, arousal);
        self
    }

    pub fn wm_accessed(mut self, at: Timestamp) -> Self {
        self.wm_accessed_at = Some(at);
        self
    }

    pub fn link(mut self, to: Id, strength: f64) -> Self {
        self.outgoing.insert(to, strength.clamp(0.0, 1.0));
        self
    }

    pub fn context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    pub fn add_link(&mut self, link: &Link) {
        if link.from == self.id {
            self.outgoing.insert(link.to, link.strength);
        }
    }

    pub fn last_access(&self) -> Option<Timestamp> {
        self.accesses.iter().copied().max()
    }
}

pub struct Builder {
    id: Id,
    embedding: Embedding,
    accesses: Vec<Timestamp>,
    emotion: Emotion,
    wm_accessed_at: Option<Timestamp>,
    outgoing: HashMap<Id, f64>,
    context: Option<String>,
}

impl Builder {
    pub fn new(embedding: Embedding) -> Self {
        Self {
            id: Id::new(),
            embedding,
            accesses: Vec::new(),
            emotion: Emotion::default(),
            wm_accessed_at: None,
            outgoing: HashMap::new(),
            context: None,
        }
    }

    pub fn id(mut self, id: Id) -> Self {
        self.id = id;
        self
    }

    pub fn accessed(mut self, at: Timestamp) -> Self {
        self.accesses.push(at);
        self
    }

    pub fn emotion(mut self, valence: f64, arousal: f64) -> Self {
        self.emotion = Emotion::new(valence, arousal);
        self
    }

    pub fn wm_accessed(mut self, at: Timestamp) -> Self {
        self.wm_accessed_at = Some(at);
        self
    }

    pub fn link(mut self, to: Id, strength: f64) -> Self {
        self.outgoing.insert(to, strength.clamp(0.0, 1.0));
        self
    }

    pub fn context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    pub fn build(self) -> Trace {
        Trace {
            id: self.id,
            embedding: self.embedding,
            accesses: self.accesses,
            emotion: self.emotion,
            wm_accessed_at: self.wm_accessed_at,
            outgoing: self.outgoing,
            context: self.context,
        }
    }
}
