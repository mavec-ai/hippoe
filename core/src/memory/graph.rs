use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::{Id, LinkKind, Timestamp};

#[derive(Debug, Clone)]
pub struct AssociationGraph {
    nodes: HashSet<Id>,
    edges: HashMap<Id, Vec<AssociationEdge>>,
    reverse_edges: HashMap<Id, Vec<AssociationEdge>>,
}

#[derive(Debug, Clone, Copy)]
pub struct AssociationEdge {
    pub from: Id,
    pub to: Id,
    pub strength: f64,
    pub kind: LinkKind,
    pub created_at: Timestamp,
    pub last_activated: Timestamp,
    pub activation_count: u64,
}

impl AssociationEdge {
    pub fn new(from: Id, to: Id, strength: f64, kind: LinkKind, at: Timestamp) -> Self {
        Self {
            from,
            to,
            strength: strength.clamp(0.0, 1.0),
            kind,
            created_at: at,
            last_activated: at,
            activation_count: 0,
        }
    }

    pub fn activate(&mut self, at: Timestamp) {
        self.last_activated = at;
        self.activation_count += 1;
    }

    pub fn decay(&mut self, current_time: Timestamp, decay_rate: f64) {
        let time_since_activation = current_time.saturating_sub(self.last_activated) as f64 / 1000.0;
        let decay_factor = (-decay_rate * time_since_activation).exp();
        self.strength *= decay_factor;
    }
}

impl AssociationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, id: Id) {
        self.nodes.insert(id);
    }

    pub fn remove_node(&mut self, id: Id) {
        self.nodes.remove(&id);
        self.edges.remove(&id);
        self.reverse_edges.remove(&id);

        for edges in self.edges.values_mut() {
            edges.retain(|e| e.to != id);
        }
        for edges in self.reverse_edges.values_mut() {
            edges.retain(|e| e.from != id);
        }
    }

    pub fn add_edge(&mut self, edge: AssociationEdge) {
        self.nodes.insert(edge.from);
        self.nodes.insert(edge.to);

        if let Some(existing) = self
            .edges
            .get_mut(&edge.from)
            .and_then(|edges| edges.iter_mut().find(|e| e.to == edge.to && e.kind == edge.kind))
        {
            existing.strength = (existing.strength + edge.strength).min(1.0);
            existing.activate(edge.last_activated);
        } else {
            self.edges
                .entry(edge.from)
                .or_default()
                .push(edge);
            self.reverse_edges
                .entry(edge.to)
                .or_default()
                .push(edge);
        }
    }

    pub fn remove_edge(&mut self, from: Id, to: Id, kind: LinkKind) {
        if let Some(edges) = self.edges.get_mut(&from) {
            edges.retain(|e| !(e.to == to && e.kind == kind));
        }
        if let Some(edges) = self.reverse_edges.get_mut(&to) {
            edges.retain(|e| !(e.from == from && e.kind == kind));
        }
    }

    pub fn get_edges_from(&self, id: Id) -> Vec<&AssociationEdge> {
        self.edges.get(&id).map(|e| e.iter().collect()).unwrap_or_default()
    }

    pub fn get_edges_to(&self, id: Id) -> Vec<&AssociationEdge> {
        self.reverse_edges.get(&id).map(|e| e.iter().collect()).unwrap_or_default()
    }

    pub fn get_edge(&self, from: Id, to: Id, kind: LinkKind) -> Option<&AssociationEdge> {
        self.edges
            .get(&from)
            .and_then(|edges| edges.iter().find(|e| e.to == to && e.kind == kind))
    }

    pub fn has_edge(&self, from: Id, to: Id) -> bool {
        self.edges
            .get(&from)
            .map(|edges| edges.iter().any(|e| e.to == to))
            .unwrap_or(false)
    }

    pub fn has_node(&self, id: Id) -> bool {
        self.nodes.contains(&id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|e| e.len()).sum()
    }

    pub fn decay_all(&mut self, current_time: Timestamp, decay_rate: f64) {
        for edges in self.edges.values_mut() {
            for edge in edges.iter_mut() {
                edge.decay(current_time, decay_rate);
            }
            edges.retain(|e| e.strength > 0.01);
        }

        for edges in self.reverse_edges.values_mut() {
            edges.retain(|e| e.strength > 0.01);
        }
    }

    pub fn find_path(&self, from: Id, to: Id, max_depth: usize) -> Option<Vec<Id>> {
        if !self.has_node(from) || !self.has_node(to) {
            return None;
        }

        if from == to {
            return Some(vec![from]);
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent_map = HashMap::new();

        visited.insert(from);
        queue.push_back((from, 0));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            for edge in self.get_edges_from(current) {
                if edge.to == to {
                    let mut path = vec![to];
                    let mut node = current;
                    path.push(node);

                    while let Some(&prev) = parent_map.get(&node) {
                        path.push(prev);
                        node = prev;
                    }

                    path.reverse();
                    return Some(path);
                }

                if visited.insert(edge.to) {
                    parent_map.insert(edge.to, current);
                    queue.push_back((edge.to, depth + 1));
                }
            }
        }

        None
    }

    pub fn neighbors(&self, id: Id, kind: Option<LinkKind>) -> Vec<Id> {
        let mut neighbors = HashSet::new();

        if let Some(edges) = self.edges.get(&id) {
            for edge in edges {
                if kind.is_none_or(|k| edge.kind == k) {
                    neighbors.insert(edge.to);
                }
            }
        }

        neighbors.into_iter().collect()
    }

    pub fn strongly_connected(&self, id: Id, threshold: f64) -> Vec<Id> {
        self.get_edges_from(id)
            .iter()
            .filter(|e| e.strength >= threshold)
            .map(|e| e.to)
            .collect()
    }

    pub fn spreading_activation(
        &self,
        source: Id,
        max_depth: usize,
        decay_factor: f64,
    ) -> HashMap<Id, f64> {
        let mut activations = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        activations.insert(source, 1.0);
        visited.insert(source);
        queue.push_back((source, 1.0, 0));

        while let Some((node, activation, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            for edge in self.get_edges_from(node) {
                let propagated_activation = activation * edge.strength * decay_factor;
                
                if propagated_activation > 0.01 {
                    let new_activation = activations
                        .entry(edge.to)
                        .or_insert(0.0);
                    *new_activation += propagated_activation;

                    if visited.insert(edge.to) {
                        queue.push_back((edge.to, propagated_activation, depth + 1));
                    }
                }
            }
        }

        activations
    }

    pub fn cluster_by_kind(&self, kind: LinkKind) -> Vec<Vec<Id>> {
        let mut clusters = Vec::new();
        let mut visited = HashSet::new();

        for &node in &self.nodes {
            if visited.contains(&node) {
                continue;
            }

            let mut cluster = Vec::new();
            let mut stack = vec![node];

            while let Some(current) = stack.pop() {
                if visited.insert(current) {
                    cluster.push(current);

                    for edge in self.get_edges_from(current) {
                        if edge.kind == kind && !visited.contains(&edge.to) {
                            stack.push(edge.to);
                        }
                    }
                    for edge in self.get_edges_to(current) {
                        if edge.kind == kind && !visited.contains(&edge.from) {
                            stack.push(edge.from);
                        }
                    }
                }
            }

            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }

        clusters
    }

    pub fn to_graphviz(&self) -> String {
        let mut output = String::from("digraph AssociationGraph {\n");

        for &id in &self.nodes {
            output.push_str(&format!("  \"{}\";\n", id));
        }

        for edges in self.edges.values() {
            for edge in edges {
                let label = format!("{:?} ({:.2})", edge.kind, edge.strength);
                output.push_str(&format!(
                    "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                    edge.from, edge.to, label
                ));
            }
        }

        output.push_str("}\n");
        output
    }
}

impl Default for AssociationGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::now;

    #[test]
    fn test_graph_creation() {
        let mut graph = AssociationGraph::new();
        let id = Id::new();

        graph.add_node(id);
        assert!(graph.has_node(id));
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_edge_addition() {
        let mut graph = AssociationGraph::new();
        let id1 = Id::new();
        let id2 = Id::new();
        let now = now();

        let edge = AssociationEdge::new(id1, id2, 0.8, LinkKind::Semantic, now);
        graph.add_edge(edge);

        assert_eq!(graph.edge_count(), 1);
        assert!(graph.get_edge(id1, id2, LinkKind::Semantic).is_some());
    }

    #[test]
    fn test_path_finding() {
        let mut graph = AssociationGraph::new();
        let id1 = Id::new();
        let id2 = Id::new();
        let id3 = Id::new();
        let now = now();

        graph.add_edge(AssociationEdge::new(id1, id2, 0.8, LinkKind::Semantic, now));
        graph.add_edge(AssociationEdge::new(id2, id3, 0.7, LinkKind::Semantic, now));

        let path = graph.find_path(id1, id3, 5);
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec![id1, id2, id3]);
    }

    #[test]
    fn test_spreading_activation() {
        let mut graph = AssociationGraph::new();
        let id1 = Id::new();
        let id2 = Id::new();
        let id3 = Id::new();
        let now = now();

        graph.add_edge(AssociationEdge::new(id1, id2, 0.8, LinkKind::Semantic, now));
        graph.add_edge(AssociationEdge::new(id2, id3, 0.7, LinkKind::Semantic, now));

        let activations = graph.spreading_activation(id1, 3, 0.5);

        assert!(activations.contains_key(&id1));
        assert!(activations.contains_key(&id2));
        assert!(activations.contains_key(&id3));
    }

    #[test]
    fn test_strongly_connected() {
        let mut graph = AssociationGraph::new();
        let id1 = Id::new();
        let id2 = Id::new();
        let id3 = Id::new();
        let now = now();

        graph.add_edge(AssociationEdge::new(id1, id2, 0.9, LinkKind::Semantic, now));
        graph.add_edge(AssociationEdge::new(id1, id3, 0.3, LinkKind::Semantic, now));

        let connected = graph.strongly_connected(id1, 0.5);
        assert_eq!(connected.len(), 1);
        assert!(connected.contains(&id2));
    }

    #[test]
    fn test_clustering() {
        let mut graph = AssociationGraph::new();
        let id1 = Id::new();
        let id2 = Id::new();
        let id3 = Id::new();
        let id4 = Id::new();
        let now = now();

        graph.add_edge(AssociationEdge::new(id1, id2, 0.8, LinkKind::Semantic, now));
        graph.add_edge(AssociationEdge::new(id2, id3, 0.7, LinkKind::Semantic, now));
        graph.add_edge(AssociationEdge::new(id1, id4, 0.6, LinkKind::Temporal, now));

        let semantic_clusters = graph.cluster_by_kind(LinkKind::Semantic);
        assert!(!semantic_clusters.is_empty());
    }
}
