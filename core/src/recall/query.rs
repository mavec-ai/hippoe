use crate::memory::Trace;
use crate::recall::temporal::TemporalLink;
use crate::types::{Embedding, Link, Timestamp};

#[derive(Clone)]
pub struct Query {
    pub probe: Embedding,
    pub memories: Vec<Trace>,
    pub links: Vec<Link>,
    pub temporal_links: Vec<TemporalLink>,
    pub now: Option<Timestamp>,
    pub context: Option<String>,
}

impl Query {
    pub fn new(probe: Embedding) -> Self {
        Self {
            probe,
            memories: Vec::new(),
            links: Vec::new(),
            temporal_links: Vec::new(),
            now: None,
            context: None,
        }
    }

    pub fn memories(mut self, memories: &[Trace]) -> Self {
        self.memories = memories.to_vec();
        self
    }

    pub fn add_memory(mut self, memory: Trace) -> Self {
        self.memories.push(memory);
        self
    }

    pub fn links(mut self, links: &[Link]) -> Self {
        self.links = links.to_vec();
        self
    }

    pub fn add_link(mut self, link: Link) -> Self {
        self.links.push(link);
        self
    }

    pub fn temporal_links(mut self, links: Vec<TemporalLink>) -> Self {
        self.temporal_links = links;
        self
    }

    pub fn add_temporal_link(mut self, link: TemporalLink) -> Self {
        self.temporal_links.push(link);
        self
    }

    pub fn now(mut self, timestamp: Timestamp) -> Self {
        self.now = Some(timestamp);
        self
    }

    pub fn context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }
}
