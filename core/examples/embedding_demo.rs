#[cfg(feature = "embedding-fastembed")]
use hippoe_core::FastEmbedProvider;

#[cfg(feature = "embedding-ollama")]
use hippoe_core::OllamaProvider;

#[cfg(feature = "embedding-openai")]
use hippoe_core::OpenAIProvider;

#[cfg(any(feature = "embedding-fastembed", feature = "embedding-ollama", feature = "embedding-openai"))]
use hippoe_core::{EmbeddingProvider, Hippocampus, InMemoryStorage, MemoryBuilder};

#[cfg(not(any(feature = "embedding-fastembed", feature = "embedding-ollama", feature = "embedding-openai")))]
use hippoe_core::{Hippocampus, InMemoryStorage, MemoryBuilder};

#[cfg(feature = "embedding-ollama")]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a * norm_b)).max(0.0)
    } else {
        0.0
    }
}

#[cfg(feature = "embedding-fastembed")]
async fn demo_fastembed() {
    println!("\n=== FastEmbed Demo ===\n");
    
    let provider = match FastEmbedProvider::new_default() {
        Ok(p) => p,
        Err(e) => {
            println!("Failed to initialize FastEmbed: {}", e);
            println!("Make sure FastEmbed models are available");
            return;
        }
    };
    
    println!("Provider: FastEmbed");
    println!("Model: BAAI/bge-small-en-v1.5");
    println!("Dimensions: {}\n", provider.dimensions());
    
    let texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming technology",
        "Rust is a systems programming language",
    ];
    
    println!("Generating embeddings for {} texts...\n", texts.len());
    
    match provider.embed_batch(&texts).await {
        Ok(embeddings) => {
            println!("Generated {} embeddings\n", embeddings.len());
            for (i, text) in texts.iter().enumerate() {
                println!("Text {}: \"{}\"", i + 1, text);
                println!("  Embedding dim: {}", embeddings[i].len());
                println!("  First 5 values: {:.4?}\n", &embeddings[i][..5.min(embeddings[i].len())]);
            }
            
            println!("Creating memories with real embeddings...\n");
            let storage = InMemoryStorage::new();
            let hippoe = Hippocampus::new(storage).unwrap();
            
            for (i, (text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
                let memory = MemoryBuilder::new(embedding.clone(), 1_000_000_000 + i as u64)
                    .text(text.to_string())
                    .importance(0.8)
                    .build();
                
                hippoe.memorize(memory).await.unwrap();
                println!("Memorized: \"{}\"", text);
            }
            
            println!("\nTotal memories stored: {}\n", hippoe.len());
            
            println!("=== Semantic Search Test ===\n");
            
            let query_text = "programming";
            println!("Query: \"{}\"\n", query_text);
            
            let query_embedding = provider.embed(query_text).await.unwrap();
            
            use hippoe_core::RetrievalMatch;
            
            let matches: Vec<RetrievalMatch> = hippoe.query()
                .similar_to(query_embedding)
                .max_results(3)
                .execute_with_scores()
                .await
                .unwrap();
            
            println!("Top results by semantic similarity:\n");
            for (rank, m) in matches.iter().enumerate() {
                let mem = hippoe.get(m.memory_id).await.unwrap().unwrap();
                let text = mem.content.text.as_deref().unwrap_or("?");
                println!("#{} - \"{}\"", rank + 1, text);
                println!("      Similarity: {:.4}", m.scores.similarity);
                println!("      Final Score: {:.4}\n", m.probability);
            }
            
            println!("Expected: \"Rust is a systems programming language\" should be #1\n");
        }
        Err(e) => {
            println!("Error generating embeddings: {}\n", e);
        }
    }
}

#[cfg(feature = "embedding-ollama")]
async fn demo_ollama() {
    println!("\n=== Ollama Demo with Cognitive Retrieval ===\n");
    
    let provider = OllamaProvider::default();
    
    println!("Provider: Ollama");
    println!("URL: {}", provider.base_url());
    println!("Model: {}", provider.model());
    println!("Dimensions: {}\n", provider.dimensions());
    
    let texts = [
        ("The quick brown fox jumps over the lazy dog", 0.9),
        ("Machine learning is transforming technology", 0.7),
        ("A fast auburn fox leaps over a sleepy canine", 0.8),
        ("Artificial intelligence and deep learning are revolutionizing tech", 0.6),
        ("Rust is a systems programming language focused on safety", 0.5),
    ];
    
    println!("Generating embeddings for {} texts...\n", texts.len());
    println!("Note: Requires Ollama running at localhost:11434\n");
    
    let text_refs: Vec<&str> = texts.iter().map(|(t, _)| *t).collect();
    
    match provider.embed_batch(&text_refs).await {
        Ok(embeddings) => {
            println!("Generated {} embeddings\n", embeddings.len());
            
            println!("=== Naive Cosine Similarity Test ===\n");
            
            let sim_fox = cosine_similarity(&embeddings[0], &embeddings[2]);
            println!("Fox paraphrase: {:.4} (expected HIGH)", sim_fox);
            
            let sim_ml = cosine_similarity(&embeddings[1], &embeddings[3]);
            println!("ML semantic:    {:.4} (expected HIGH)", sim_ml);
            
            let sim_unrelated = cosine_similarity(&embeddings[0], &embeddings[1]);
            println!("Unrelated:      {:.4} (expected LOW)", sim_unrelated);
            
            if sim_fox > sim_unrelated && sim_ml > sim_unrelated {
                println!("\n✅ Embedding quality: OK");
            } else {
                println!("\n❌ Embedding quality: Issue detected");
            }
            
            println!("\n=== Cognitive Memory System ===\n");
            
            let storage = InMemoryStorage::new();
            let hippoe = Hippocampus::new(storage).unwrap();
            
            for (i, ((text, importance), embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
                let memory = MemoryBuilder::new(embedding.clone(), 1_000_000_000 + i as u64)
                    .text(text.to_string())
                    .importance(*importance)
                    .build();
                
                hippoe.memorize(memory).await.unwrap();
                println!("Memorized [importance={}]: \"{}\"", importance, text);
            }
            
            println!("\nTotal memories: {}\n", hippoe.len());
            
            println!("=== Cognitive Retrieval Test ===\n");
            
            let query = "programming languages";
            println!("Query: \"{}\"\n", query);
            
            let query_embedding = provider.embed(query).await.unwrap();
            
            let rust_idx = texts.iter().position(|(t, _)| t.contains("Rust")).unwrap();
            let naive_score = cosine_similarity(&query_embedding, &embeddings[rust_idx]);
            
            use hippoe_core::RetrievalMatch;
            
            let matches: Vec<RetrievalMatch> = hippoe.query()
                .similar_to(query_embedding)
                .max_results(5)
                .execute_with_scores()
                .await
                .unwrap();
            
            println!("Results ranked by Cognitive HybridStrategy:\n");
            println!("{:-<60}", "");
            for (rank, m) in matches.iter().enumerate() {
                let mem = hippoe.get(m.memory_id).await.unwrap().unwrap();
                let text = mem.content.text.as_deref().unwrap_or("?");
                let imp = mem.metadata.importance;
                
                println!("#{} - \"{}\"", rank + 1, if text.len() > 50 { &text[..50] } else { text });
                println!("     Similarity:   {:.4}", m.scores.similarity);
                println!("     Base-level:   {:.4}", m.scores.base_level);
                println!("     Importance:   {:.2}", imp);
                println!("     Final Score:  {:.4}", m.probability);
                println!("{:-<60}", "");
            }
            
            println!("\n=== Cognitive vs Naive Comparison ===\n");
            
            let mut rust_match: Option<&RetrievalMatch> = None;
            for m in matches.iter() {
                let mem = hippoe.get(m.memory_id).await.unwrap().unwrap();
                if mem.content.text.as_deref().unwrap_or("").contains("Rust") {
                    rust_match = Some(m);
                    break;
                }
            }
            
            println!("Query: \"{}\"", query);
            println!("Target: \"Rust is a systems programming language...\"\n");
            println!("Naive cosine similarity:  {:.4}", naive_score);
            
            if let Some(rust) = rust_match {
                println!("Cognitive final score:   {:.4}", rust.probability);
                println!("Similarity component:    {:.4}", rust.scores.similarity);
                println!("Base-level activation:   {:.4}", rust.scores.base_level);
                
                if rust.probability > naive_score {
                    println!("\n✅ Cognitive retrieval boosted the relevant result!");
                }
            }
            
            println!("\n=== Cognitive Components Active ===");
            println!("• Similarity weight:  1.0");
            println!("• Base-level weight:  0.8 (frequency/recency)");
            println!("• Spreading weight:   0.7 (associative activation)");
            println!("• Emotional weight:   0.5 (importance/arousal)");
            println!("• Contextual weight:  0.3");
            println!("• Temporal weight:    0.3\n");
        }
        Err(e) => {
            println!("Error: {}\n", e);
            println!("Make sure Ollama is running: ollama serve");
            println!("And the model is pulled: ollama pull nomic-embed-text\n");
        }
    }
}

#[cfg(feature = "embedding-openai")]
fn cosine_similarity_openai(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a * norm_b)).max(0.0)
    } else {
        0.0
    }
}

#[cfg(feature = "embedding-openai")]
async fn demo_openai() {
    println!("\n=== OpenAI Demo with Cognitive Retrieval ===\n");

    let provider = match std::env::var("OPENAI_API_KEY") {
        Ok(_) => OpenAIProvider::default(),
        Err(_) => {
            println!("❌ OPENAI_API_KEY environment variable not set\n");
            println!("Set it with:");
            println!("  export OPENAI_API_KEY='your-api-key'\n");
            println!("Or use:");
            println!("  let provider = OpenAIProvider::default()");
            println!("      .with_api_key(\"your-api-key\");\n");
            return;
        }
    };

    println!("Provider: OpenAI");
    println!("Model: {}", provider.model_name());
    println!("Dimensions: {}\n", provider.dimensions());

    let texts = [
        ("The quick brown fox jumps over the lazy dog", 0.9),
        ("Machine learning is transforming technology", 0.7),
        ("A fast auburn fox leaps over a sleepy canine", 0.8),
        ("Artificial intelligence and deep learning are revolutionizing tech", 0.6),
        ("Rust is a systems programming language focused on safety", 0.5),
    ];

    println!("Generating embeddings for {} texts...\n", texts.len());

    let text_refs: Vec<&str> = texts.iter().map(|(t, _)| *t).collect();

    match provider.embed_batch(&text_refs).await {
        Ok(embeddings) => {
            println!("Generated {} embeddings\n", embeddings.len());

            println!("=== Naive Cosine Similarity Test ===\n");

            let sim_fox = cosine_similarity_openai(&embeddings[0], &embeddings[2]);
            println!("Fox paraphrase: {:.4} (expected HIGH)", sim_fox);

            let sim_ml = cosine_similarity_openai(&embeddings[1], &embeddings[3]);
            println!("ML semantic:    {:.4} (expected HIGH)", sim_ml);

            let sim_unrelated = cosine_similarity_openai(&embeddings[0], &embeddings[1]);
            println!("Unrelated:      {:.4} (expected LOW)", sim_unrelated);

            if sim_fox > sim_unrelated && sim_ml > sim_unrelated {
                println!("\n✅ Embedding quality: OK");
            } else {
                println!("\n❌ Embedding quality: Issue detected");
            }

            println!("\n=== Cognitive Memory System ===\n");

            let storage = InMemoryStorage::new();
            let hippoe = Hippocampus::new(storage).unwrap();

            for (i, ((text, importance), embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
                let memory = MemoryBuilder::new(embedding.clone(), 1_000_000_000 + i as u64)
                    .text(text.to_string())
                    .importance(*importance)
                    .build();

                hippoe.memorize(memory).await.unwrap();
                println!("Memorized [importance={}]: \"{}\"", importance, text);
            }

            println!("\nTotal memories: {}\n", hippoe.len());

            println!("=== Cognitive Retrieval Test ===\n");

            let query = "programming languages";
            println!("Query: \"{}\"\n", query);

            let query_embedding = provider.embed(query).await.unwrap();

            let rust_idx = texts.iter().position(|(t, _)| t.contains("Rust")).unwrap();
            let naive_score = cosine_similarity_openai(&query_embedding, &embeddings[rust_idx]);

            use hippoe_core::RetrievalMatch;

            let matches: Vec<RetrievalMatch> = hippoe.query()
                .similar_to(query_embedding)
                .max_results(5)
                .execute_with_scores()
                .await
                .unwrap();

            println!("Results ranked by Cognitive HybridStrategy:\n");
            println!("{:-<60}", "");
            for (rank, m) in matches.iter().enumerate() {
                let mem = hippoe.get(m.memory_id).await.unwrap().unwrap();
                let text = mem.content.text.as_deref().unwrap_or("?");
                let imp = mem.metadata.importance;

                println!("#{} - \"{}\"", rank + 1, if text.len() > 50 { &text[..50] } else { text });
                println!("     Similarity:   {:.4}", m.scores.similarity);
                println!("     Base-level:   {:.4}", m.scores.base_level);
                println!("     Importance:   {:.2}", imp);
                println!("     Final Score:  {:.4}", m.probability);
                println!("{:-<60}", "");
            }

            println!("\n=== Cognitive vs Naive Comparison ===\n");

            let mut rust_match: Option<&RetrievalMatch> = None;
            for m in matches.iter() {
                let mem = hippoe.get(m.memory_id).await.unwrap().unwrap();
                if mem.content.text.as_deref().unwrap_or("").contains("Rust") {
                    rust_match = Some(m);
                    break;
                }
            }

            println!("Query: \"{}\"", query);
            println!("Target: \"Rust is a systems programming language...\"\n");
            println!("Naive cosine similarity:  {:.4}", naive_score);

            if let Some(rust) = rust_match {
                println!("Cognitive final score:   {:.4}", rust.probability);
                println!("Similarity component:    {:.4}", rust.scores.similarity);
                println!("Base-level activation:   {:.4}", rust.scores.base_level);

                if rust.probability > naive_score {
                    println!("\n✅ Cognitive retrieval boosted the relevant result!");
                }
            }

            println!("\n=== Cognitive Components Active ===");
            println!("• Similarity weight:  1.0");
            println!("• Base-level weight:  0.8 (frequency/recency)");
            println!("• Spreading weight:   0.7 (associative activation)");
            println!("• Emotional weight:   0.5 (importance/arousal)");
            println!("• Contextual weight:  0.3");
            println!("• Temporal weight:    0.3\n");

            println!("=== Model Options ===");
            println!("• text-embedding-3-small: 1536 dim (default, cost-effective)");
            println!("• text-embedding-3-large: 3072 dim (highest quality)");
            println!("• text-embedding-ada-002: 1536 dim (legacy)\n");
        }
        Err(e) => {
            println!("Error: {}\n", e);
            println!("Make sure OPENAI_API_KEY is valid and has credits.\n");
        }
    }
}

#[cfg(not(any(feature = "embedding-fastembed", feature = "embedding-ollama", feature = "embedding-openai")))]
fn demo_without_embedding(rt: &tokio::runtime::Runtime) {
    println!("\n=== Demo with Fake Embeddings ===\n");
    println!("No embedding feature enabled. Using fake embeddings.\n");
    println!("To enable real embeddings, use one of:");
    println!("  cargo run --example embedding_demo --features embedding-fastembed");
    println!("  cargo run --example embedding_demo --features embedding-ollama");
    println!("  cargo run --example embedding_demo --features embedding-openai\n");
    
    let storage = InMemoryStorage::new();
    let hippoe = Hippocampus::new(storage).unwrap();
    
    let texts = vec![
        "The quick brown fox",
        "Machine learning basics",
        "Rust programming",
    ];
    
    fn fake_embedding(dim: usize, seed: f64) -> Vec<f64> {
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
    
    for (i, text) in texts.iter().enumerate() {
        let embedding = fake_embedding(384, i as f64 * 0.3);
        let memory = MemoryBuilder::new(embedding, 1_000_000_000 + i as u64)
            .text(text.to_string())
            .build();

        rt.block_on(hippoe.memorize(memory)).unwrap();
    }
    
    println!("Created {} memories with fake embeddings\n", hippoe.len());
}

fn main() {
    println!("=== Hippoe Embedding Providers Demo ===");

    let rt = tokio::runtime::Runtime::new().unwrap();
    
    #[cfg(feature = "embedding-fastembed")]
    rt.block_on(demo_fastembed());
    
    #[cfg(feature = "embedding-ollama")]
    rt.block_on(demo_ollama());
    
    #[cfg(feature = "embedding-openai")]
    rt.block_on(demo_openai());
    
    #[cfg(not(any(feature = "embedding-fastembed", feature = "embedding-ollama", feature = "embedding-openai")))]
    demo_without_embedding(&rt);
    
    println!("=== Demo Complete ===\n");
}
