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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: &[f64]) -> Embedding {
        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            values.iter().map(|x| x / norm).collect()
        } else {
            values.to_vec()
        }
    }

    #[test]
    fn test_trace_builder() {
        let id = Id::new();
        let embedding = make_embedding(&[1.0, 2.0, 3.0]);

        let trace = Trace::new(id, embedding.clone())
            .accessed(1000)
            .accessed(2000)
            .emotion(0.8, 0.6);

        assert_eq!(trace.id, id);
        assert_eq!(trace.accesses.len(), 2);
        assert_eq!(trace.emotion.valence, 0.8);
        assert_eq!(trace.emotion.arousal, 0.6);
    }

    #[test]
    fn test_trace_linking() {
        let id1 = Id::new();
        let id2 = Id::new();
        let embedding = make_embedding(&[1.0, 0.0, 0.0]);

        let trace = Trace::new(id1, embedding).link(id2, 0.5);

        assert_eq!(trace.outgoing.len(), 1);
        assert_eq!(*trace.outgoing.get(&id2).unwrap(), 0.5);
    }
}
