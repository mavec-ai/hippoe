#![allow(clippy::expect_used)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hippoe_core::types::{LinkKind, now};
use hippoe_core::{Id, InMemoryStorage, Memory, MemoryBuilder, Storage};
use rand::Rng;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

#[cfg(feature = "sqlite")]
use hippoe_core::SqliteStorage;

fn generate_memories(count: usize, dimensions: usize) -> Vec<Memory> {
    let mut rng = rand::thread_rng();
    let ts = now();
    (0..count)
        .map(|_| {
            let embedding: Vec<f64> = (0..dimensions).map(|_| rng.gen_range(0.0..1.0)).collect();
            MemoryBuilder::new(embedding, ts)
                .text(format!("memory_{}", Id::new()))
                .build()
        })
        .collect()
}

fn bench_storage_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_put");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000, 2000] {
        let memories = generate_memories(count, 1536);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |bench, _| {
            bench.iter(|| {
                let storage = InMemoryStorage::new();
                rt.block_on(async {
                    for memory in memories.clone() {
                        storage.put(memory).await.unwrap();
                    }
                });
                black_box(storage)
            });
        });
    }

    group.finish();
}

fn bench_storage_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_get");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000, 2000] {
        let memories = generate_memories(count, 1536);
        let ids: Vec<Id> = memories.iter().map(|m| m.id).collect();

        let storage = InMemoryStorage::new();
        rt.block_on(async {
            for memory in memories {
                storage.put(memory).await.unwrap();
            }
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |bench, _| {
            bench.iter(|| {
                rt.block_on(async {
                    for id in &ids {
                        black_box(storage.get(*id).await.unwrap());
                    }
                });
            });
        });
    }

    group.finish();
}

fn bench_storage_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_all");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000, 2000] {
        let memories = generate_memories(count, 1536);

        let storage = InMemoryStorage::new();
        rt.block_on(async {
            for memory in memories {
                storage.put(memory).await.unwrap();
            }
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |bench, _| {
            bench.iter(|| {
                rt.block_on(async {
                    black_box(storage.all().await.unwrap());
                });
            });
        });
    }

    group.finish();
}

fn bench_storage_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_remove");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000, 2000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |bench, _| {
            bench.iter(|| {
                let memories = generate_memories(count, 1536);
                let ids: Vec<Id> = memories.iter().map(|m| m.id).collect();

                let storage = InMemoryStorage::new();
                rt.block_on(async {
                    for memory in memories {
                        storage.put(memory).await.unwrap();
                    }
                });

                rt.block_on(async {
                    for id in &ids {
                        let _: () = storage.remove(*id).await.unwrap();
                        black_box(());
                    }
                });
            });
        });
    }

    group.finish();
}

fn generate_memories_with_links(
    count: usize,
    dimensions: usize,
    links_per_memory: usize,
) -> Vec<Memory> {
    let mut rng = rand::thread_rng();
    let ids: Vec<Id> = (0..count).map(|_| Id::new()).collect();
    let ts = now();

    ids.iter()
        .map(|&id| {
            let embedding: Vec<f64> = (0..dimensions).map(|_| rng.gen_range(0.0..1.0)).collect();

            let mut builder = MemoryBuilder::new(embedding, ts)
                .id(id)
                .text(format!("memory_{}", id));

            for &link_id in ids.iter().take(std::cmp::min(links_per_memory, count)) {
                if id != link_id {
                    builder = builder.link(link_id, 0.5, LinkKind::Semantic, ts);
                }
            }
            builder.build()
        })
        .collect()
}

fn bench_storage_links(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_links");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000] {
        let memories = generate_memories_with_links(count, 1536, 5);

        let storage = InMemoryStorage::new();
        rt.block_on(async {
            for memory in memories {
                storage.put(memory).await.unwrap();
            }
        });

        let link_count = count * std::cmp::min(5, count);
        group.throughput(Throughput::Elements(link_count as u64));
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |bench, _| {
            bench.iter(|| {
                rt.block_on(async {
                    let graph = storage.get_graph().await.unwrap();
                    black_box(graph.edge_count());
                });
            });
        });
    }

    group.finish();
}

fn bench_storage_mixed_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_mixed");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000] {
        let memories = generate_memories(count * 2, 1536);
        let ids: Vec<Id> = memories.iter().map(|m| m.id).collect();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("operations", count), &count, |bench, _| {
            bench.iter(|| {
                let storage = InMemoryStorage::new();
                rt.block_on(async {
                    for memory in memories.iter().take(count) {
                        storage.put(memory.clone()).await.unwrap();
                    }

                    for id in ids.iter().take(count / 2) {
                        storage.get(*id).await.unwrap();
                    }

                    for id in ids.iter().take(count / 4) {
                        storage.remove(*id).await.unwrap();
                    }

                    storage.all().await.unwrap();
                });
                black_box(storage)
            });
        });
    }

    group.finish();
}

fn bench_storage_concurrent_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_concurrent_reads");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000] {
        let memories = generate_memories(count, 1536);
        let ids: Vec<Id> = memories.iter().map(|m| m.id).collect();

        let storage = Arc::new(Mutex::new(InMemoryStorage::new()));
        rt.block_on(async {
            let s = storage.lock().await;
            for memory in memories {
                s.put(memory).await.unwrap();
            }
        });

        let storage_clone = Arc::clone(&storage);
        let ids_clone = ids.clone();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("concurrent", count), &count, |bench, _| {
            bench.iter(|| {
                let storage = Arc::clone(&storage_clone);
                let ids = ids_clone.clone();
                rt.block_on(async {
                    let s = storage.lock().await;
                    for id in &ids {
                        black_box(s.get(*id).await.unwrap());
                    }
                });
            });
        });
    }

    group.finish();
}

fn bench_storage_concurrent_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_concurrent_writes");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000] {
        let memories = generate_memories(count, 1536);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("concurrent", count), &count, |bench, _| {
            bench.iter(|| {
                let storage = Arc::new(Mutex::new(InMemoryStorage::new()));
                let memories = memories.clone();
                rt.block_on(async {
                    let s = storage.lock().await;
                    for memory in memories {
                        s.put(memory).await.unwrap();
                    }
                });
            });
        });
    }

    group.finish();
}

fn bench_storage_large_dataset(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_large_dataset");
    group.sample_size(10);
    let rt = Runtime::new().unwrap();

    for &count in &[5000, 10000] {
        let memories = generate_memories(count, 1536);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("insert", count), &count, |bench, _| {
            bench.iter(|| {
                let storage = InMemoryStorage::new();
                rt.block_on(async {
                    for memory in memories.clone() {
                        storage.put(memory).await.unwrap();
                    }
                });
                black_box(storage)
            });
        });
    }

    group.finish();
}

fn bench_storage_large_dataset_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_large_dataset_read");
    group.sample_size(10);
    let rt = Runtime::new().unwrap();

    for &count in &[5000, 10000] {
        let memories = generate_memories(count, 1536);
        let ids: Vec<Id> = memories.iter().map(|m| m.id).collect();

        let storage = InMemoryStorage::new();
        rt.block_on(async {
            for memory in memories {
                storage.put(memory).await.unwrap();
            }
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("read_all", count), &count, |bench, _| {
            bench.iter(|| {
                rt.block_on(async {
                    for id in &ids {
                        black_box(storage.get(*id).await.unwrap());
                    }
                });
            });
        });
    }

    group.finish();
}

#[cfg(feature = "sqlite")]
fn bench_sqlite_storage_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_storage_put");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000] {
        let memories = generate_memories(count, 1536);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |bench, _| {
            bench.iter(|| {
                let storage = rt.block_on(async {
                    let s = SqliteStorage::new_in_memory().await.unwrap();
                    for memory in memories.clone() {
                        s.put(memory).await.unwrap();
                    }
                    s
                });
                black_box(storage)
            });
        });
    }

    group.finish();
}

#[cfg(feature = "sqlite")]
fn bench_sqlite_storage_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_storage_get");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000] {
        let memories = generate_memories(count, 1536);
        let ids: Vec<Id> = memories.iter().map(|m| m.id).collect();

        let storage = rt.block_on(async {
            let s = SqliteStorage::new_in_memory().await.unwrap();
            for memory in memories {
                s.put(memory).await.unwrap();
            }
            s
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |bench, _| {
            bench.iter(|| {
                rt.block_on(async {
                    for id in &ids {
                        black_box(storage.get(*id).await.unwrap());
                    }
                });
            });
        });
    }

    group.finish();
}

#[cfg(feature = "sqlite")]
fn bench_sqlite_storage_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_storage_all");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000] {
        let memories = generate_memories(count, 1536);

        let storage = rt.block_on(async {
            let s = SqliteStorage::new_in_memory().await.unwrap();
            for memory in memories {
                s.put(memory).await.unwrap();
            }
            s
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |bench, _| {
            bench.iter(|| {
                rt.block_on(async {
                    black_box(storage.all().await.unwrap());
                });
            });
        });
    }

    group.finish();
}

#[cfg(feature = "sqlite")]
fn bench_sqlite_vs_inmemory(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_vs_inmemory");
    let rt = Runtime::new().unwrap();

    let count = 500;
    let memories = generate_memories(count, 1536);
    let ids: Vec<Id> = memories.iter().map(|m| m.id).collect();

    group.throughput(Throughput::Elements(count as u64));

    group.bench_function("inmemory_put", |bench| {
        bench.iter(|| {
            let storage = InMemoryStorage::new();
            rt.block_on(async {
                for memory in memories.clone() {
                    storage.put(memory).await.unwrap();
                }
            });
            black_box(storage)
        });
    });

    group.bench_function("sqlite_put", |bench| {
        bench.iter(|| {
            let storage = rt.block_on(async {
                let s = SqliteStorage::new_in_memory().await.unwrap();
                for memory in memories.clone() {
                    s.put(memory).await.unwrap();
                }
                s
            });
            black_box(storage)
        });
    });

    let inmem_storage = InMemoryStorage::new();
    rt.block_on(async {
        for memory in memories.clone() {
            inmem_storage.put(memory).await.unwrap();
        }
    });

    let sqlite_storage = rt.block_on(async {
        let s = SqliteStorage::new_in_memory().await.unwrap();
        for memory in memories.clone() {
            s.put(memory).await.unwrap();
        }
        s
    });

    group.bench_function("inmemory_get_all", |bench| {
        bench.iter(|| {
            rt.block_on(async {
                for id in &ids {
                    black_box(inmem_storage.get(*id).await.unwrap());
                }
            });
        });
    });

    group.bench_function("sqlite_get_all", |bench| {
        bench.iter(|| {
            rt.block_on(async {
                for id in &ids {
                    black_box(sqlite_storage.get(*id).await.unwrap());
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    storage_benches,
    bench_storage_put,
    bench_storage_get,
    bench_storage_all,
    bench_storage_remove,
    bench_storage_links,
    bench_storage_mixed_operations,
    bench_storage_concurrent_reads,
    bench_storage_concurrent_writes,
    bench_storage_large_dataset,
    bench_storage_large_dataset_read,
);

#[cfg(feature = "sqlite")]
criterion_group!(
    sqlite_benches,
    bench_sqlite_storage_put,
    bench_sqlite_storage_get,
    bench_sqlite_storage_all,
    bench_sqlite_vs_inmemory,
);

#[cfg(not(feature = "sqlite"))]
criterion_main!(storage_benches);

#[cfg(feature = "sqlite")]
criterion_main!(storage_benches, sqlite_benches);
