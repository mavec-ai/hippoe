use hippoe_core::{
    Hippocampus, InMemoryStorage, MemoryBuilder,
    LinkKind,
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
    println!("=== Hippoe Spreading Activation Demo ===\n");

    let storage = InMemoryStorage::new();
    let hippoe: Hippocampus<InMemoryStorage> = Hippocampus::new(storage).unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();

    let memory_data = vec![
        ("Rust ownership rules", &[1.0, 0.0, 0.2, 0.0][..]),
        ("Memory safety in systems", &[0.8, 0.3, 0.1, 0.0][..]),
        ("WebAssembly compilation", &[0.3, 0.0, 0.9, 0.1][..]),
        ("API design patterns", &[0.0, 0.2, 0.1, 1.0][..]),
        ("Async runtime internals", &[0.5, 0.8, 0.0, 0.3][..]),
    ];

    println!("Creating memory network...\n");

    let memory_ids: Vec<_> = rt.block_on(async {
        let mut ids = Vec::new();
        for (text, embedding_vals) in &memory_data {
            let embedding = generate_embedding(64, embedding_vals);
            let memory = MemoryBuilder::new(embedding, 1_000_000_000_u64)
                .text(*text)
                .importance(0.5)
                .build();

            let id = hippoe.memorize(memory).await.unwrap();
            ids.push(id);
            println!("  [{}] {}", id, text);
        }
        ids
    });

    println!("\nBuilding associations...\n");

    rt.block_on(async {
        let rust_ownership = memory_ids[0];
        let memory_safety = memory_ids[1];
        let wasm = memory_ids[2];
        let api_patterns = memory_ids[3];
        let async_runtime = memory_ids[4];

        let associations = vec![
            (rust_ownership, memory_safety, 0.9, "Rust ownership", "Memory safety"),
            (rust_ownership, wasm, 0.6, "Rust ownership", "WebAssembly"),
            (memory_safety, async_runtime, 0.5, "Memory safety", "Async runtime"),
            (wasm, api_patterns, 0.7, "WebAssembly", "API patterns"),
            (async_runtime, api_patterns, 0.8, "Async runtime", "API patterns"),
        ];

        for (from, to, strength, from_text, to_text) in associations {
            hippoe.create_association(from, to, strength, LinkKind::Semantic)
                .await
                .unwrap();
            println!("  {} --({:.1})--> {}", from_text, strength, to_text);
        }
    });

    println!("\n--- Query WITHOUT spreading (depth=0) ---\n");

    rt.block_on(async {
        let query_embedding = generate_embedding(64, &[1.0, 0.0, 0.2, 0.0]);
        let results = hippoe.query()
            .similar_to(query_embedding)
            .max_results(5)
            .execute()
            .await
            .unwrap();

        for (rank, mem) in results.iter().enumerate() {
            println!("#{} - {}", rank + 1, mem.content.text.as_deref().unwrap_or("?"));
        }
    });

    println!("\n--- Query WITH associations ---\n");

    rt.block_on(async {
        let query_embedding = generate_embedding(64, &[1.0, 0.0, 0.2, 0.0]);
        let rust_id = memory_ids[0];
        let results = hippoe.query()
            .similar_to(query_embedding)
            .max_results(5)
            .include_associations(rust_id, 3)
            .execute()
            .await
            .unwrap();

        println!("Results including associations from 'Rust ownership rules':\n");
        for (rank, mem) in results.iter().enumerate() {
            println!("#{} - {}", rank + 1, mem.content.text.as_deref().unwrap_or("?"));
        }
    });

    println!("\n--- Graph Statistics ---\n");

    rt.block_on(async {
        let graph = hippoe.get_graph().await.unwrap();
        println!("Nodes: {}", graph.node_count());
        println!("Edges: {}", graph.edge_count());
    });

    println!("\n=== Demo Complete ===");
}
