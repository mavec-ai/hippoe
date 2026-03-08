pub mod types;
pub mod config;
pub mod error;
pub mod hippocampus;
pub mod memory;
pub mod recall;
pub mod decay;

pub use types::{Embedding, Emotion, Id, Link, LinkKind, Timestamp};
pub use config::Config;
pub use hippocampus::Hippocampus;
pub use memory::Trace;
pub use recall::Query;
pub use decay::{time_decay, boost, Curve, history_score};
pub use recall::scorer::{similarity, similarity_batch};

#[cfg(test)]
fn make_embedding(values: &[f64]) -> Embedding {
    let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        values.iter().map(|x| x / norm).collect()
    } else {
        values.to_vec()
    }
}

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
fn test_time_decay() {
    let now: Timestamp = 100000;
    
    let decay_recent = time_decay(90000, now, 0.5);
    let decay_old = time_decay(50000, now, 0.5);
    
    assert!(decay_recent > decay_old);
    assert!(decay_recent <= 1.0);
    assert!(decay_old >= 0.0);
    
    let decay_future = time_decay(110000, now, 0.5);
    assert_eq!(decay_future, 1.0);
}

#[test]
fn test_boost() {
    let now: Timestamp = 100000;
    
    let boost_recent = boost(90000, now, 2.0);
    let boost_old = boost(50000, now, 2.0);
    
    assert!(boost_recent > boost_old);
    assert!(boost_recent <= 2.0);
    
    let boost_future = boost(110000, now, 2.0);
    assert_eq!(boost_future, 2.0);
}

#[test]
fn test_curve_exponential() {
    let curve = Curve::Exponential { rate: 0.5 };
    let now: Timestamp = 100000;
    
    let decay_recent = curve.decay(90000, now);
    let decay_old = curve.decay(50000, now);
    
    assert!(decay_recent > decay_old);
}

#[test]
fn test_curve_power_law() {
    let curve = Curve::PowerLaw { exponent: 0.8 };
    let now: Timestamp = 100000;
    
    let decay_recent = curve.decay(90000, now);
    let decay_old = curve.decay(50000, now);
    
    assert!(decay_recent > decay_old);
}

#[test]
fn test_curve_linear() {
    let curve = Curve::Linear { slope: 0.0001 };
    let now: Timestamp = 100000;
    
    let decay_recent = curve.decay(90000, now);
    let decay_old = curve.decay(50000, now);
    
    assert!(decay_recent > decay_old);
}

#[test]
fn test_similarity_identical() {
    let a = make_embedding(&[1.0, 0.0, 0.0]);
    let b = make_embedding(&[1.0, 0.0, 0.0]);
    
    let sim = similarity(&a, &b);
    assert!(sim > 0.99);
}

#[test]
fn test_similarity_orthogonal() {
    let a = make_embedding(&[1.0, 0.0, 0.0]);
    let b = make_embedding(&[0.0, 1.0, 0.0]);
    
    let sim = similarity(&a, &b);
    assert!(sim < 0.01);
}

#[test]
fn test_similarity_opposite() {
    let a = make_embedding(&[1.0, 0.0, 0.0]);
    let b = make_embedding(&[-1.0, 0.0, 0.0]);
    
    let sim = similarity(&a, &b);
    assert!(sim < 0.01);
}

#[test]
fn test_similarity_batch() {
    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let t1 = make_embedding(&[1.0, 0.0, 0.0]);
    let t2 = make_embedding(&[0.0, 1.0, 0.0]);
    let t3 = make_embedding(&[0.707, 0.707, 0.0]);
    
    let targets: Vec<&[f64]> = vec![&t1, &t2, &t3];
    
    let sims = similarity_batch(&probe, &targets);
    
    assert_eq!(sims.len(), 3);
    assert!(sims[0] > 0.99);
    assert!(sims[1] < 0.01);
    assert!(sims[2] > 0.3 && sims[2] < 0.5);
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

#[test]
fn test_config_builder() {
    let config = Config::builder()
        .decay_rate(0.3)
        .spread_depth(3)
        .min_score(0.05)
        .max_results(20)
        .boost_cap(1.5)
        .emotion_weight(0.2)
        .build()
        .unwrap();
    
    assert_eq!(config.decay_rate, 0.3);
    assert_eq!(config.spread_depth, 3);
    assert_eq!(config.min_score, 0.05);
    assert_eq!(config.max_results, 20);
    assert_eq!(config.boost_cap, 1.5);
    assert_eq!(config.emotion_weight, 0.2);
}

#[test]
fn test_hippocampus_default() {
    let hippoe = Hippocampus::new();
    let config = hippoe.config();
    
    assert!(config.decay_rate > 0.0);
    assert!(config.spread_depth > 0);
}

#[test]
fn test_recall_basic() {
    let hippoe = Hippocampus::new();
    
    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let mem1 = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]))
        .accessed(1000);
    let mem2 = Trace::new(Id::new(), make_embedding(&[0.1, 0.9, 0.0]))
        .accessed(1000);
    
    let query = Query::new(probe)
        .add_memory(mem1)
        .add_memory(mem2)
        .now(2000);
    
    let result = hippoe.recall(query).unwrap();
    
    assert_eq!(result.len(), 2);
    assert!(result.matches()[0].score.similarity > result.matches()[1].score.similarity);
}

#[test]
fn test_recall_with_spreading() {
    let hippoe = Hippocampus::builder()
        .spread_depth(2)
        .build()
        .unwrap();

    let id1 = Id::new();
    let id2 = Id::new();
    let id3 = Id::new();
    
    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let mem1 = Trace::new(id1, make_embedding(&[0.9, 0.1, 0.0]));
    let mem2 = Trace::new(id2, make_embedding(&[0.1, 0.9, 0.0]));
    let mem3 = Trace::new(id3, make_embedding(&[0.2, 0.8, 0.0]));
    
    let link = Link::semantic(id1, id2, 0.8);
    
    let query = Query::new(probe)
        .add_memory(mem1)
        .add_memory(mem2)
        .add_memory(mem3)
        .add_link(link)
        .now(1000);
    
    let result = hippoe.recall(query).unwrap();
    
    assert_eq!(result.len(), 3);
}

#[test]
fn test_recall_empty_probe_error() {
    let hippoe = Hippocampus::new();
    
    let query = Query::new(Vec::new());
    
    let result = hippoe.recall(query);
    assert!(result.is_err());
}

#[test]
fn test_recall_no_memories_error() {
    let hippoe = Hippocampus::new();
    
    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let query = Query::new(probe);
    
    let result = hippoe.recall(query);
    assert!(result.is_err());
}

#[test]
fn test_recall_dimension_mismatch_error() {
    let hippoe = Hippocampus::new();
    
    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let mem = Trace::new(Id::new(), make_embedding(&[1.0, 0.0, 0.0, 0.0]));
    
    let query = Query::new(probe).add_memory(mem);
    
    let result = hippoe.recall(query);
    assert!(result.is_err());
}

#[test]
fn test_recall_with_emotion() {
    let hippoe = Hippocampus::builder()
        .emotion_weight(0.5)
        .build()
        .unwrap();

    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let mem_neutral = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]));
    let mem_emotional = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]))
        .emotion(0.9, 0.9);
    
    let query = Query::new(probe.clone())
        .add_memory(mem_neutral.clone())
        .add_memory(mem_emotional.clone())
        .now(1000);
    
    let result = hippoe.recall(query).unwrap();
    
    let emotional_idx = result.matches().iter()
        .position(|m| m.id == mem_emotional.id);
    
    if let Some(idx) = emotional_idx {
        assert!(result.matches()[idx].score.emotion > 1.0);
    }
}

#[test]
fn test_recall_result_methods() {
    let hippoe = Hippocampus::new();
    
    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let mem1 = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]));
    let mem2 = Trace::new(Id::new(), make_embedding(&[0.8, 0.2, 0.0]));
    let mem3 = Trace::new(Id::new(), make_embedding(&[0.1, 0.9, 0.0]));
    
    let query = Query::new(probe)
        .add_memory(mem1)
        .add_memory(mem2)
        .add_memory(mem3)
        .now(1000);
    
    let result = hippoe.recall(query).unwrap();
    
    assert_eq!(result.len(), 3);
    assert!(!result.is_empty());
    
    let top2 = result.top(2);
    assert_eq!(top2.len(), 2);
    
    let first = result.first();
    assert!(first.is_some());
    
    let ids = result.ids();
    assert_eq!(ids.len(), 3);
}

#[test]
fn test_recall_with_working_memory() {
    let hippoe = Hippocampus::builder()
        .boost_cap(2.0)
        .build()
        .unwrap();
    
    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let mem_wm = Trace::new(Id::new(), make_embedding(&[0.7, 0.3, 0.0]))
        .wm_accessed(900);
    let mem_no_wm = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]));
    
    let query = Query::new(probe)
        .add_memory(mem_wm)
        .add_memory(mem_no_wm)
        .now(1000);
    
    let result = hippoe.recall(query).unwrap();
    
    assert!(result.matches().iter().any(|m| m.score.boost > 1.0));
}

#[test]
fn test_recall_probability_sums() {
    let hippoe = Hippocampus::new();
    
    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let mem1 = Trace::new(Id::new(), make_embedding(&[0.9, 0.1, 0.0]));
    let mem2 = Trace::new(Id::new(), make_embedding(&[0.8, 0.2, 0.0]));
    let mem3 = Trace::new(Id::new(), make_embedding(&[0.7, 0.3, 0.0]));
    
    let query = Query::new(probe)
        .add_memory(mem1)
        .add_memory(mem2)
        .add_memory(mem3)
        .now(1000);
    
    let result = hippoe.recall(query).unwrap();
    
    let prob_sum: f64 = result.matches().iter().map(|m| m.probability).sum();
    assert!((prob_sum - 1.0).abs() < 0.001);
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

#[test]
fn test_min_score_filtering() {
    let hippoe = Hippocampus::builder()
        .min_score(0.5)
        .build()
        .unwrap();

    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let mem_high = Trace::new(Id::new(), make_embedding(&[0.99, 0.01, 0.0]));
    let mem_low = Trace::new(Id::new(), make_embedding(&[0.0, 0.99, 0.0]));
    
    let query = Query::new(probe)
        .add_memory(mem_high)
        .add_memory(mem_low)
        .now(1000);
    
    let result = hippoe.recall(query).unwrap();
    
    for m in result.matches() {
        assert!(m.score.total >= 0.5);
    }
}

#[test]
fn test_max_results_limit() {
    let hippoe = Hippocampus::builder()
        .max_results(2)
        .build()
        .unwrap();

    let probe = make_embedding(&[1.0, 0.0, 0.0]);
    let memories: Vec<Trace> = (0..10)
        .map(|i| Trace::new(Id::new(), make_embedding(&[0.9 - i as f64 * 0.05, 0.1, 0.0])))
        .collect();
    
    let mut query = Query::new(probe);
    for m in memories {
        query = query.add_memory(m);
    }
    query = query.now(1000);
    
    let result = hippoe.recall(query).unwrap();
    
    assert!(result.len() <= 2);
}
