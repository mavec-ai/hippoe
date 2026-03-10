use hippoe_core::{
    HippocampusBuilder,
    LinkKind,
    memory::MemoryBuilder,
    storage::InMemoryStorage,
};

fn generate_embedding(dim: usize, values: &[f64]) -> Vec<f64> {
    let mut embedding = Vec::with_capacity(dim);
    for i in 0..dim {
        let idx = i % values.len();
        let val = values[idx] + (i as f64 * 0.001);
        embedding.push(val);
    }
    let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        embedding.iter().map(|x| x / norm).collect()
    } else {
        embedding
    }
}

fn main() {
    println!("=== Hippoe Cognitive Hippocampus Full Demo ===\n");

    let rt = tokio::runtime::Runtime::new().unwrap();

    println!("1. Creating Hippocampus with custom config...\n");

    let storage = InMemoryStorage::new();
    let hippoe = HippocampusBuilder::default()
        .decay_rate(0.1)
        .min_score(0.1)
        .max_results(10)
        .emotion_weight(0.3)
        .context_weight(0.2)
        .semantic_threshold(0.5)
        .build(storage)
        .unwrap();

    println!("Configuration:");
    println!("  Decay rate: {}", hippoe.config().decay_rate);
    println!("  Min score: {}", hippoe.config().min_score);
    println!("  Max results: {}", hippoe.config().max_results);
    println!("  Emotion weight: {}", hippoe.config().emotion_weight);
    println!("  Context weight: {}", hippoe.config().context_weight);

    let now = 1_000_000_000_u64;

    println!("\n2. Storing memories with emotional weight...\n");

    let memory_ids: Vec<_> = rt.block_on(async {
        let memories_data = vec![
            ("Birthday party celebration", 0.9, 0.9, vec!["personal", "happy"]),
            ("Project deadline reminder", 0.7, 0.6, vec!["work", "stress"]),
            ("Team building event", 0.5, 0.8, vec!["work", "social"]),
            ("Important presentation", 0.8, 0.7, vec!["work"]),
            ("Weekend hiking trip", 0.6, 0.85, vec!["personal", "outdoor"]),
        ];

        let mut ids = Vec::new();
        for (text, importance, emotion, tags) in &memories_data {
            let embedding = generate_embedding(64, &[*importance, *emotion]);
            let mut builder = MemoryBuilder::new(embedding, now)
                .text(*text)
                .importance(*importance)
                .emotion(*emotion - 0.1, *emotion);

            for tag in tags {
                builder = builder.tag(*tag);
            }

            let memory = builder.build();
            let id = hippoe.memorize(memory).await.unwrap();
            ids.push(id);

            println!("  '{}' (importance: {:.1}, emotion: {:.1})", text, importance, emotion);
        }
        ids
    });

    println!("\n3. Creating associations between memories...\n");

    rt.block_on(async {
        let birthday = memory_ids[0];
        let coffee = memory_ids[2];
        let hiking = memory_ids[4];

        hippoe.create_association(birthday, coffee, 0.8, LinkKind::Semantic)
            .await.unwrap();
        hippoe.create_association(coffee, hiking, 0.6, LinkKind::Episodic)
            .await.unwrap();
        hippoe.create_association(birthday, hiking, 0.5, LinkKind::Semantic)
            .await.unwrap();

        println!("  Birthday <-> Team building (0.8)");
        println!("  Team building <-> Hiking (0.6)");
        println!("  Birthday <-> Hiking (0.5)");
    });

    println!("\n4. Retrieving with spreading activation...\n");

    rt.block_on(async {
        let probe = generate_embedding(64, &[0.9, 0.9]);
        let matches = hippoe.recall(probe).await.unwrap();

        for (rank, m) in matches.iter().enumerate() {
            let mem = hippoe.get(m.memory_id).await.unwrap().unwrap();
            println!("#{} - {} (prob: {:.4})", 
                rank + 1,
                mem.content.text.as_deref().unwrap_or("?"),
                m.probability
            );
            println!("     Similarity: {:.4}, Base Level: {:.4}, Spreading: {:.4}",
                m.scores.similarity,
                m.scores.base_level,
                m.scores.spreading
            );
        }
    });

    println!("\n5. Querying associated memories...\n");

    rt.block_on(async {
        let birthday_id = memory_ids[0];
        let associated = hippoe.recall_associated(birthday_id, 2).await.unwrap();

        println!("Memories associated with 'Birthday party':");
        for mem in &associated {
            println!("  - {}", mem.content.text.as_deref().unwrap_or("?"));
        }
    });

    println!("\n6. Graph statistics...\n");

    rt.block_on(async {
        let graph = hippoe.get_graph().await.unwrap();
        println!("Total memories: {}", hippoe.len());
        println!("Graph nodes: {}", graph.node_count());
        println!("Graph edges: {}", graph.edge_count());
    });

    println!("\n7. Reconsolidation (importance boost on recall)...\n");

    rt.block_on(async {
        let before = hippoe.get(memory_ids[0]).await.unwrap().unwrap();
        let importance_before = before.metadata.importance;

        let probe = generate_embedding(64, &[0.9, 0.9]);
        let _ = hippoe.recall(probe).await.unwrap();

        let after = hippoe.get(memory_ids[0]).await.unwrap().unwrap();
        let importance_after = after.metadata.importance;

        println!("Importance before recall: {:.4}", importance_before);
        println!("Importance after recall: {:.4}", importance_after);
        println!("Boost: {:.2}%", (importance_after / importance_before - 1.0) * 100.0);
    });

    println!("\n8. Forget (delete) a memory...\n");

    rt.block_on(async {
        let count_before = hippoe.len();
        println!("Memories before forget: {}", count_before);

        hippoe.forget(memory_ids[3]).await.unwrap();
        println!("Forgot: 'Important presentation'");

        let count_after = hippoe.len();
        println!("Memories after forget: {}", count_after);
    });

    println!("\n=== Demo Complete ===");
}
