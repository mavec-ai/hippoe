#![allow(clippy::expect_used)]

use std::collections::HashMap;

use hippoe_core::{
    AssociationGraph, Embedding, HippocampusBuilder, Id, InMemoryStorage, Memory, MemoryBuilder,
    RetrievalContext, RetrievalMatch, RetrievalStrategy, Timestamp, similarity_batch,
};
use uuid::Uuid;

fn make_deterministic_id(index: usize) -> Id {
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&(index as u64).to_le_bytes());
    bytes[8..16].copy_from_slice(&[0xB0, 0xB0, 0xC0, 0xC0, 0xD0, 0xD0, 0xE0, 0xE0]);
    Id(Uuid::from_bytes_le(bytes))
}

fn make_embedding(pattern: &[f64], dim: usize) -> Embedding {
    let mut emb = vec![0.0; dim];
    for (i, &p) in pattern.iter().enumerate().take(dim) {
        emb[i] = p;
    }
    let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut emb {
            *x /= norm;
        }
    }
    emb
}

fn current_time() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as Timestamp
}

struct TestCase {
    name: &'static str,
    description: &'static str,
    probe: Embedding,
    memories: Vec<TestMemory>,
    associations: Vec<TestAssociation>,
    current_time: Timestamp,
}

struct TestMemory {
    embedding: Embedding,
    importance: f64,
    emotional_weight: f64,
    access_count: u32,
    session_accesses: u32,
    other_session_accesses: bool,
    expected_rank: usize,
}

impl Default for TestMemory {
    fn default() -> Self {
        Self {
            embedding: vec![],
            importance: 0.5,
            emotional_weight: 0.5,
            access_count: 1,
            session_accesses: 0,
            other_session_accesses: false,
            expected_rank: 1,
        }
    }
}

struct TestAssociation {
    source_idx: usize,
    target_idx: usize,
    strength: f64,
}

struct QualityMetrics {
    ndcg: f64,
    mrr: f64,
    precision_at_k: f64,
}

fn compute_ndcg(rankings: &[usize], k: usize) -> f64 {
    let k = k.min(rankings.len());

    let max_rank = rankings.iter().copied().max().unwrap_or(1);
    let relevance: Vec<f64> = rankings
        .iter()
        .map(|&rank| (max_rank - rank + 1) as f64)
        .collect();

    let dcg: f64 = relevance
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| (2.0_f64.powf(rel) - 1.0) / ((i + 2) as f64).log2())
        .sum();

    let mut ideal_relevance = relevance.clone();
    ideal_relevance.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f64 = ideal_relevance
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| (2.0_f64.powf(rel) - 1.0) / ((i + 2) as f64).log2())
        .sum();

    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

fn compute_mrr(rankings: &[usize]) -> f64 {
    if rankings.is_empty() {
        return 0.0;
    }
    1.0 / rankings[0] as f64
}

fn compute_precision_at_k(rankings: &[usize], k: usize, threshold: usize) -> f64 {
    let k = k.min(rankings.len());
    let relevant = rankings.iter().take(k).filter(|&&r| r <= threshold).count();
    relevant as f64 / k as f64
}

struct NaiveStrategy;

#[async_trait::async_trait]
impl RetrievalStrategy for NaiveStrategy {
    async fn retrieve(
        &self,
        memories: &[Memory],
        _graph: &AssociationGraph,
        context: &RetrievalContext,
    ) -> Vec<RetrievalMatch> {
        let targets: Vec<&[f64]> = memories.iter().map(|m| m.embedding.as_slice()).collect();
        let similarities = similarity_batch(&context.query_embedding, &targets);

        let mut results: Vec<(usize, f64)> = similarities
            .iter()
            .enumerate()
            .map(|(i, &sim)| (i, sim))
            .collect();

        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results.truncate(context.max_results);

        results
            .into_iter()
            .map(|(idx, sim)| RetrievalMatch {
                memory_id: memories[idx].id,
                scores: hippoe_core::RetrievalScores {
                    similarity: sim,
                    total: sim,
                    ..Default::default()
                },
                probability: (sim + 1.0) / 2.0,
            })
            .collect()
    }
}

fn get_test_cases() -> Vec<TestCase> {
    let dim = 128;
    let now = current_time();

    vec![
        TestCase {
            name: "recency_tiebreaker",
            description: "Two identical memories, one accessed recently. Cognitive should break the tie.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[1.0, 0.0, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[1.0, 0.0, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 0,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "frequency_over_similarity",
            description: "Frequently accessed memory should beat slightly-more-similar infrequent one.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.9, 0.1, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 20,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.95, 0.05, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "emotional_boost",
            description: "Emotionally significant memory should rank higher (equal similarity, equal recency).",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.9, 0.1, 0.0], dim),
                    emotional_weight: 1.0,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.9, 0.1, 0.0], dim),
                    emotional_weight: 0.0,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "spreading_surfaces_related",
            description: "Associated memory should rank higher than unassociated one.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[1.0, 0.0, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.2, 0.8, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.5, 0.5, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 3,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![TestAssociation {
                source_idx: 0,
                target_idx: 1,
                strength: 0.95,
            }],
        },
        TestCase {
            name: "combined_signals",
            description: "Multiple weak signals should beat single strong signal.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.85, 0.15, 0.0], dim),
                    emotional_weight: 0.9,
                    importance: 0.8,
                    access_count: 5,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.95, 0.05, 0.0], dim),
                    emotional_weight: 0.3,
                    importance: 0.3,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "base_level_frequency",
            description: "Frequently accessed memory (base_level) should rank higher.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.8, 0.2, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 10,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.9, 0.1, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "session_working_memory",
            description: "Recently accessed in CURRENT session should boost (working memory).",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.85, 0.15, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 5,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.95, 0.05, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "session_isolation",
            description: "Working memory boost should be session-specific.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.9, 0.1, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.85, 0.15, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 10,
                    other_session_accesses: true,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "pattern_completion_30_percent",
            description: "Retrieve memory from 30% corrupted probe (MINERVA 2 style).",
            probe: {
                let mut emb = make_embedding(&[1.0, 0.0, 0.0], dim);
                for (i, val) in emb.iter_mut().enumerate() {
                    if i % 3 == 0 {
                        *val *= -1.0;
                    }
                }
                emb
            },
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[1.0, 0.0, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.0, 1.0, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "fan_effect",
            description: "Target with same direct similarity but low-fan source should outrank high-fan source target (ACT-R fan effect).",
            probe: make_embedding(&[0.9, 0.1, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.7, 0.3, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.7, 0.3, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.5, 0.5, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 5,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.5, 0.5, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 6,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.2, 0.3, 0.5], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 3,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.15, 0.35, 0.5], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 4,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.1, 0.4, 0.5], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 7,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.25, 0.25, 0.5], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 8,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![
                TestAssociation {
                    source_idx: 2,
                    target_idx: 0,
                    strength: 0.95,
                },
                TestAssociation {
                    source_idx: 2,
                    target_idx: 4,
                    strength: 0.9,
                },
                TestAssociation {
                    source_idx: 2,
                    target_idx: 5,
                    strength: 0.85,
                },
                TestAssociation {
                    source_idx: 2,
                    target_idx: 6,
                    strength: 0.8,
                },
                TestAssociation {
                    source_idx: 3,
                    target_idx: 1,
                    strength: 0.95,
                },
            ],
        },
        TestCase {
            name: "temporal_contiguity",
            description: "TCM-style: temporally adjacent memories should be linked.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[1.0, 0.0, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.3, 0.7, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.0, 0.0, 1.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 3,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![
                TestAssociation {
                    source_idx: 0,
                    target_idx: 1,
                    strength: 0.9,
                },
                TestAssociation {
                    source_idx: 1,
                    target_idx: 2,
                    strength: 0.8,
                },
            ],
        },
        TestCase {
            name: "multiplicative_over_additive",
            description: "Multiplicative scoring should prevent low-similarity recent memory from beating high-similarity old one.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.95, 0.05, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.5, 0.5, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 50,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "retrieval_probability_filtering",
            description: "Memory with very low probability should be filtered out (not just low similarity).",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[1.0, 0.0, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.9, 0.1, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.1, 0.9, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 3,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "contextual_temporal_normalization",
            description: "Contextual and temporal scores should be clamped to [0,1] for consistent scoring.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[0.9, 0.1, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.9,
                    access_count: 5,
                    expected_rank: 1,
                    session_accesses: 3,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.85, 0.15, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![],
        },
        TestCase {
            name: "probe_activation_dominance",
            description: "Probe activation (similarity cubed) should dominate over weak spreading activation.",
            probe: make_embedding(&[1.0, 0.0, 0.0], dim),
            current_time: now,
            memories: vec![
                TestMemory {
                    embedding: make_embedding(&[1.0, 0.0, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 1,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.3, 0.7, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 3,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
                TestMemory {
                    embedding: make_embedding(&[0.7, 0.3, 0.0], dim),
                    emotional_weight: 0.5,
                    importance: 0.5,
                    access_count: 1,
                    expected_rank: 2,
                    session_accesses: 0,
                    other_session_accesses: false,
                },
            ],
            associations: vec![TestAssociation {
                source_idx: 1,
                target_idx: 2,
                strength: 0.9,
            }],
        },
    ]
}

fn build_memories_from_test(test: &TestCase) -> Vec<Memory> {
    test.memories
        .iter()
        .enumerate()
        .map(|(i, tm)| {
            let created_at = test.current_time - 3_600_000;
            let mut memory = MemoryBuilder::new(tm.embedding.clone(), created_at)
                .id(make_deterministic_id(i))
                .text(format!("Memory {}", i))
                .importance(tm.importance)
                .emotion(tm.emotional_weight, tm.emotional_weight)
                .decay_rate(0.0001)
                .build();

            for _ in 0..tm.access_count {
                memory.accessed(test.current_time - 60_000);
            }

            memory
        })
        .collect()
}

async fn run_naive_retrieval(test: &TestCase, memories: &[Memory]) -> Vec<usize> {
    let graph = AssociationGraph::new();
    let strategy = NaiveStrategy;
    let context = RetrievalContext::new(test.probe.clone(), test.current_time)
        .with_max_results(memories.len());

    let results = strategy.retrieve(memories, &graph, &context).await;

    let mut id_to_expected: HashMap<_, _> = memories
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id, test.memories[i].expected_rank))
        .collect();

    results
        .iter()
        .map(|r| id_to_expected.remove(&r.memory_id).unwrap_or(999))
        .collect()
}

async fn run_cognitive_retrieval(test: &TestCase, memories: &[Memory]) -> Vec<usize> {
    let hippoe = HippocampusBuilder::default()
        .build(InMemoryStorage::new())
        .expect("Failed to build hippocampus");

    hippoe.set_session("benchmark_session");

    let mut memory_ids = Vec::new();
    for memory in memories.iter() {
        let id = memory.id;
        hippoe
            .memorize(memory.clone())
            .await
            .expect("Failed to memorize");
        memory_ids.push(id);
    }

    for assoc in &test.associations {
        let from = memory_ids[assoc.source_idx];
        let to = memory_ids[assoc.target_idx];
        hippoe
            .create_association(from, to, assoc.strength, hippoe_core::LinkKind::Semantic)
            .await
            .expect("Failed to create association");
    }

    for (idx, tm) in test.memories.iter().enumerate() {
        if tm.session_accesses > 0 {
            let mem_id = memory_ids[idx];
            for _ in 0..tm.session_accesses {
                hippoe
                    .working_memory()
                    .record_access(mem_id, "benchmark_session");
            }
        }
        if tm.other_session_accesses {
            let mem_id = memory_ids[idx];
            for _ in 0..10 {
                hippoe
                    .working_memory()
                    .record_access(mem_id, "other_session");
            }
        }
    }

    let results = hippoe
        .recall(test.probe.clone())
        .await
        .expect("Failed to recall");

    let mut id_to_expected: HashMap<_, _> = memory_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, test.memories[i].expected_rank))
        .collect();

    results
        .iter()
        .map(|r| id_to_expected.remove(&r.memory_id).unwrap_or(999))
        .collect()
}

fn evaluate_rankings(rankings: &[usize], k: usize) -> QualityMetrics {
    QualityMetrics {
        ndcg: compute_ndcg(rankings, k),
        mrr: compute_mrr(rankings),
        precision_at_k: compute_precision_at_k(rankings, k, 3),
    }
}

fn main() {
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create runtime");

    let test_cases = get_test_cases();
    let mut naive_wins = 0;
    let mut cognitive_wins = 0;
    let mut ties = 0;
    let mut total_naive_ndcg = 0.0;
    let mut total_cognitive_ndcg = 0.0;

    println!("=== Hippoe Quality Benchmark: Naive vs Cognitive ===\n");

    for test in &test_cases {
        let memories = build_memories_from_test(test);

        let naive_rankings = runtime.block_on(run_naive_retrieval(test, &memories));
        let cognitive_rankings = runtime.block_on(run_cognitive_retrieval(test, &memories));

        let naive_metrics = evaluate_rankings(&naive_rankings, 5);
        let cognitive_metrics = evaluate_rankings(&cognitive_rankings, 5);

        total_naive_ndcg += naive_metrics.ndcg;
        total_cognitive_ndcg += cognitive_metrics.ndcg;

        let winner = if cognitive_metrics.ndcg > naive_metrics.ndcg {
            cognitive_wins += 1;
            "COGNITIVE"
        } else if naive_metrics.ndcg > cognitive_metrics.ndcg {
            naive_wins += 1;
            "NAIVE"
        } else {
            ties += 1;
            "TIE"
        };

        println!("=== Test: {} ===", test.name);
        println!("  {}", test.description);
        println!(
            "  Naive:     NDCG={:.4}, MRR={:.4}, P@5={:.4}",
            naive_metrics.ndcg, naive_metrics.mrr, naive_metrics.precision_at_k
        );
        println!(
            "  Cognitive: NDCG={:.4}, MRR={:.4}, P@5={:.4}",
            cognitive_metrics.ndcg, cognitive_metrics.mrr, cognitive_metrics.precision_at_k
        );
        println!("  Winner: {}\n", winner);
    }

    let test_count = test_cases.len() as f64;
    let avg_naive_ndcg = total_naive_ndcg / test_count;
    let avg_cognitive_ndcg = total_cognitive_ndcg / test_count;
    let improvement = ((avg_cognitive_ndcg - avg_naive_ndcg) / avg_naive_ndcg) * 100.0;

    println!("=== Summary ===");
    println!("Cognitive wins: {}/{}", cognitive_wins, test_cases.len());
    println!("Naive wins:     {}/{}", naive_wins, test_cases.len());
    println!("Ties:           {}/{}", ties, test_cases.len());
    println!();
    println!("Average NDCG:");
    println!("  Naive:     {:.4}", avg_naive_ndcg);
    println!("  Cognitive: {:.4}", avg_cognitive_ndcg);
    println!("  Improvement: {:+.1}%", improvement);
}
