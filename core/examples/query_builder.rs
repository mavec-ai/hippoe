use hippoe_core::{CognitiveRetrieval, Hippocampus, InMemoryStorage, MemoryBuilder};

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
    println!("=== Hippoe Query Builder Demo ===\n");

    let storage = InMemoryStorage::new();
    let hippoe = Hippocampus::new(storage).unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let now = 1_000_000_000_u64;

    println!("Creating sample memories...\n");

    rt.block_on(async {
        let memories_data = vec![
            ("Rust async programming", 0.9, vec!["programming", "rust"]),
            ("Python data analysis", 0.8, vec!["programming", "python"]),
            ("Morning coffee ritual", 0.6, vec!["lifestyle"]),
            ("Important project deadline", 0.95, vec!["work", "deadline"]),
            ("Team meeting notes", 0.7, vec!["work", "meeting"]),
        ];

        for (text, importance, tags) in &memories_data {
            let embedding = generate_embedding(64, &[*importance, 0.5]);
            let mut builder = MemoryBuilder::new(embedding, now)
                .text(*text)
                .importance(*importance);

            for tag in tags {
                builder = builder.tag(*tag);
            }

            let memory = builder.build();
            hippoe.memorize(memory).await.unwrap();
            println!("  Created: '{}' (importance: {:.2})", text, importance);
        }
    });

    println!("\n=== Query Examples ===\n");

    println!("1. Basic similarity query:\n");
    rt.block_on(async {
        let probe = generate_embedding(64, &[0.9, 0.5]);
        let results = hippoe
            .query()
            .similar_to(probe)
            .max_results(3)
            .execute()
            .await
            .unwrap();

        for (i, mem) in results.iter().enumerate() {
            println!(
                "  #{} - {}",
                i + 1,
                mem.content.text.as_deref().unwrap_or("?")
            );
        }
    });

    println!("\n2. Filter by tag:\n");
    rt.block_on(async {
        let results = hippoe
            .query()
            .with_tag("work")
            .max_results(5)
            .execute()
            .await
            .unwrap();

        for (i, mem) in results.iter().enumerate() {
            println!(
                "  #{} - {} [tags: {:?}]",
                i + 1,
                mem.content.text.as_deref().unwrap_or("?"),
                mem.metadata.tags
            );
        }
    });

    println!("\n3. Filter by minimum importance:\n");
    rt.block_on(async {
        let results = hippoe
            .query()
            .with_min_importance(0.8)
            .max_results(5)
            .execute()
            .await
            .unwrap();

        for (i, mem) in results.iter().enumerate() {
            println!(
                "  #{} - {} (importance: {:.2})",
                i + 1,
                mem.content.text.as_deref().unwrap_or("?"),
                mem.metadata.importance
            );
        }
    });

    println!("\n4. Combined filters:\n");
    rt.block_on(async {
        let probe = generate_embedding(64, &[0.85, 0.5]);
        let results = hippoe
            .query()
            .similar_to(probe)
            .with_tag("programming")
            .min_similarity(0.0)
            .max_results(5)
            .execute()
            .await
            .unwrap();

        for (i, mem) in results.iter().enumerate() {
            println!(
                "  #{} - {}",
                i + 1,
                mem.content.text.as_deref().unwrap_or("?")
            );
        }
    });

    println!("\n5. With custom retrieval strategy (CognitiveRetrieval):\n");
    rt.block_on(async {
        let probe = generate_embedding(64, &[0.9, 0.5]);
        let strategy = CognitiveRetrieval::new().with_emotional_weight(0.6);

        let results = hippoe
            .query()
            .similar_to(probe)
            .with_strategy(Box::new(strategy))
            .max_results(3)
            .execute_with_scores()
            .await
            .unwrap();

        for (i, m) in results.iter().enumerate() {
            let mem = hippoe.get(m.memory_id).await.unwrap().unwrap();
            println!(
                "  #{} - {} (prob: {:.4})",
                i + 1,
                mem.content.text.as_deref().unwrap_or("?"),
                m.probability
            );
        }
    });

    println!("\n=== Demo Complete ===");
}
