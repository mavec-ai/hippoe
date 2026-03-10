#![allow(clippy::expect_used)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hippoe_core::{Hippocampus, InMemoryStorage, MemoryBuilder, similarity, similarity_batch};
use rand::Rng;

fn generate_embeddings(count: usize, dimensions: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let mut vec: Vec<f64> = (0..dimensions).map(|_| rng.gen_range(0.0..1.0)).collect();
            let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut vec {
                    *x /= norm;
                }
            }
            vec
        })
        .collect()
}

fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    for dim in &[128, 256, 512, 1024, 1536] {
        let embeddings = generate_embeddings(2, *dim);
        let a = &embeddings[0];
        let b = &embeddings[1];

        let _ = group.throughput(Throughput::Elements(1));
        let _ = group.bench_with_input(BenchmarkId::new("single", dim), dim, |bench, _| {
            bench.iter(|| black_box(similarity(black_box(a), black_box(b))));
        });
    }

    group.finish();
}

fn bench_similarity_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_batch");

    for &(memory_count, dim) in &[(100, 1024), (500, 1024), (1000, 1024), (2000, 1024)] {
        let probe = generate_embeddings(1, dim).pop().unwrap();
        let memories = generate_embeddings(memory_count, dim);
        let targets: Vec<&[f64]> = memories.iter().map(|m| m.as_slice()).collect();

        let _ = group.throughput(Throughput::Elements(memory_count as u64));
        let _ = group.bench_with_input(
            BenchmarkId::new("memories", memory_count),
            &memory_count,
            |bench, _| {
                bench.iter(|| black_box(similarity_batch(black_box(&probe), black_box(&targets))));
            },
        );
    }

    group.finish();
}

fn bench_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall");

    for &(memory_count, dim) in &[(100, 512), (500, 512), (1000, 512), (2000, 512)] {
        let storage = InMemoryStorage::new();
        let hippoe = Hippocampus::new(storage).unwrap();
        let now = 1_000_000_000_u64;

        let probe = generate_embeddings(1, dim).pop().unwrap();
        let embeddings = generate_embeddings(memory_count, dim);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            for embedding in embeddings {
                let memory = MemoryBuilder::new(embedding, now).build();
                hippoe.memorize(memory).await.unwrap();
            }
        });

        let _ = group.throughput(Throughput::Elements(memory_count as u64));
        let _ = group.bench_with_input(
            BenchmarkId::new("memories", memory_count),
            &memory_count,
            |bench, _| {
                bench.iter(|| black_box(rt.block_on(hippoe.recall(black_box(probe.clone())))));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_similarity,
    bench_similarity_batch,
    bench_recall,
);

criterion_main!(benches);
