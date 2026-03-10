use hippoe_core::{Hippocampus, InMemoryStorage, MemoryBuilder};

fn generate_embedding(dim: usize, seed: f64) -> Vec<f64> {
    let mut embedding = Vec::with_capacity(dim);
    for i in 0..dim {
        let val = (seed * (i + 1) as f64 * 0.1).sin() * 0.5 + 0.5;
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
    println!("=== Hippoe Basic Memory Demo ===\n");

    let storage = InMemoryStorage::new();
    let hippoe = Hippocampus::new(storage).unwrap();

    let now = 1_000_000_000_u64;

    let memories = vec![
        ("Rust programming language", 0.9, 0.1),
        ("Python machine learning", 0.2, 0.8),
        ("Morning coding session", 0.7, 0.3),
        ("Database optimization", 0.4, 0.6),
        ("Web API development", 0.5, 0.5),
    ];

    println!("Storing {} memories...\n", memories.len());

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        for (text, seed1, seed2) in &memories {
            let embedding = generate_embedding(128, *seed1 * *seed2);
            let memory = MemoryBuilder::new(embedding, now)
                .text(*text)
                .importance(*seed1)
                .build();

            hippoe.memorize(memory).await.unwrap();
            println!("  Stored: '{}'", text);
        }
    });

    println!("\n--- Query: programming ---\n");

    rt.block_on(async {
        let query_embedding = generate_embedding(128, 0.85);
        let results = hippoe
            .query()
            .similar_to(query_embedding)
            .max_results(3)
            .execute()
            .await
            .unwrap();

        for (rank, mem) in results.iter().enumerate() {
            println!(
                "#{} - {}",
                rank + 1,
                mem.content.text.as_deref().unwrap_or("?")
            );
            println!("    Importance: {:.2}", mem.metadata.importance);
            println!(
                "    Created: {} ms ago",
                now.saturating_sub(mem.metadata.created_at)
            );
            println!();
        }
    });

    println!("--- Recall by tag ---\n");

    rt.block_on(async {
        let tagged_memory = MemoryBuilder::new(generate_embedding(128, 0.6), now)
            .text("Important meeting notes")
            .importance(0.9)
            .tag("work")
            .tag("meeting")
            .build();

        hippoe.memorize(tagged_memory).await.unwrap();

        let results = hippoe
            .query()
            .with_tags(vec!["work".to_string()])
            .max_results(5)
            .execute()
            .await
            .unwrap();

        println!("Found {} memories with tag 'work':\n", results.len());
        for mem in &results {
            println!("  - {}", mem.content.text.as_deref().unwrap_or("?"));
            println!("    Tags: {:?}", mem.metadata.tags);
        }
    });

    println!("\n=== Demo Complete ===");
}
