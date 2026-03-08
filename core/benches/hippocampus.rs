#![allow(clippy::expect_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hippoe_core::{similarity, similarity_batch, time_decay, boost, decay::history_score, Hippocampus, Trace, Query, Id};
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

fn bench_time_decay(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_decay");
    let now = 1_000_000_000_u64;
    let rate = 0.5;

    for &delta_ms in &[1000, 10_000, 100_000, 1_000_000, 10_000_000] {
        let last_access = now.saturating_sub(delta_ms);

        let _ = group.bench_with_input(
            BenchmarkId::new("delta_ms", delta_ms),
            &delta_ms,
            |bench, _| {
                bench.iter(|| black_box(time_decay(black_box(last_access), black_box(now), black_box(rate))));
            },
        );
    }

    group.finish();
}

fn bench_boost(c: &mut Criterion) {
    let mut group = c.benchmark_group("boost");
    let now = 1_000_000_000_u64;
    let cap = 2.0;

    for &delta_ms in &[1000, 10_000, 100_000, 1_000_000] {
        let accessed_at = now.saturating_sub(delta_ms);

        let _ = group.bench_with_input(
            BenchmarkId::new("delta_ms", delta_ms),
            &delta_ms,
            |bench, _| {
                bench.iter(|| black_box(boost(black_box(accessed_at), black_box(now), black_box(cap))));
            },
        );
    }

    group.finish();
}

fn bench_history_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("history_score");
    let now = 1_000_000_000_u64;
    let rate = 0.5;

    for &access_count in &[5, 10, 20, 50, 100] {
        let mut rng = rand::thread_rng();
        let accesses: Vec<u64> = (0..access_count)
            .map(|_| now.saturating_sub(rng.gen_range(0..604_800_000)))
            .collect();

        let _ = group.bench_with_input(
            BenchmarkId::new("accesses", access_count),
            &access_count,
            |bench, _| {
                bench.iter(|| black_box(history_score(black_box(&accesses), black_box(now), black_box(rate))));
            },
        );
    }

    group.finish();
}

fn bench_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall");

    for &(memory_count, dim) in &[(100, 512), (500, 512), (1000, 512), (2000, 512)] {
        let hippoe = Hippocampus::new();
        let now = 1_000_000_000_u64;

        let probe = generate_embeddings(1, dim).pop().unwrap();
        let embeddings = generate_embeddings(memory_count, dim);

        let mut query = Query::new(probe);
        for embedding in embeddings {
            query = query.add_memory(
                Trace::new(Id::new(), embedding).accessed(now.saturating_sub(100_000))
            );
        }
        query = query.now(now);

        let _ = group.throughput(Throughput::Elements(memory_count as u64));
        let _ = group.bench_with_input(
            BenchmarkId::new("memories", memory_count),
            &memory_count,
            |bench, _| {
                bench.iter(|| black_box(hippoe.recall(black_box(query.clone()))));
            },
        );
    }

    group.finish();
}

fn bench_recall_with_spreading(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_with_spreading");

    let hippoe = Hippocampus::builder().spread_depth(2).build().unwrap();
    let dim = 512;
    let now = 1_000_000_000_u64;

    for &memory_count in &[100, 500, 1000] {
        let probe = generate_embeddings(1, dim).pop().unwrap();
        let embeddings = generate_embeddings(memory_count, dim);

        let mut query = Query::new(probe);
        for embedding in embeddings {
            query = query.add_memory(
                Trace::new(Id::new(), embedding).accessed(now.saturating_sub(100_000))
            );
        }
        query = query.now(now);

        let _ = group.throughput(Throughput::Elements(memory_count as u64));
        let _ = group.bench_with_input(
            BenchmarkId::new("memories", memory_count),
            &memory_count,
            |bench, _| {
                bench.iter(|| black_box(hippoe.recall(black_box(query.clone()))));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_similarity,
    bench_similarity_batch,
    bench_time_decay,
    bench_boost,
    bench_history_score,
    bench_recall,
    bench_recall_with_spreading,
);

criterion_main!(benches);
