#![allow(clippy::expect_used)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hippoe_core::{Id, InMemoryStorage, Storage, Trace};
use rand::Rng;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

#[cfg(feature = "sqlite")]
use hippoe_core::SqliteStorage;

fn generate_traces(count: usize, dimensions: usize) -> Vec<Trace> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let embedding: Vec<f64> = (0..dimensions).map(|_| rng.gen_range(0.0..1.0)).collect();
            Trace::new(Id::new(), embedding)
        })
        .collect()
}

fn bench_storage_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_put");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000, 2000] {
        let traces = generate_traces(count, 1536);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("traces", count), &count, |bench, _| {
            bench.iter(|| {
                let mut storage = InMemoryStorage::new();
                rt.block_on(async {
                    for trace in traces.clone() {
                        black_box(storage.put(trace).await.unwrap());
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
        let traces = generate_traces(count, 1536);
        let ids: Vec<Id> = traces.iter().map(|t| t.id).collect();

        let mut storage = InMemoryStorage::new();
        rt.block_on(async {
            for trace in traces {
                storage.put(trace).await.unwrap();
            }
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("traces", count), &count, |bench, _| {
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
        let traces = generate_traces(count, 1536);

        let mut storage = InMemoryStorage::new();
        rt.block_on(async {
            for trace in traces {
                storage.put(trace).await.unwrap();
            }
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("traces", count), &count, |bench, _| {
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
        group.bench_with_input(BenchmarkId::new("traces", count), &count, |bench, _| {
            bench.iter(|| {
                let traces = generate_traces(count, 1536);
                let ids: Vec<Id> = traces.iter().map(|t| t.id).collect();

                let mut storage = InMemoryStorage::new();
                rt.block_on(async {
                    for trace in traces {
                        storage.put(trace).await.unwrap();
                    }
                });

                rt.block_on(async {
                    for id in &ids {
                        black_box(storage.remove(*id).await.unwrap());
                    }
                });
            });
        });
    }

    group.finish();
}

fn generate_traces_with_links(
    count: usize,
    dimensions: usize,
    links_per_trace: usize,
) -> Vec<Trace> {
    let mut rng = rand::thread_rng();
    let ids: Vec<Id> = (0..count).map(|_| Id::new()).collect();

    ids.iter()
        .enumerate()
        .map(|(i, &id)| {
            let embedding: Vec<f64> = (0..dimensions).map(|_| rng.gen_range(0.0..1.0)).collect();

            let mut trace = Trace::new(id, embedding);
            for j in 0..std::cmp::min(links_per_trace, count) {
                if i != j {
                    trace = trace.link(ids[j], 0.5);
                }
            }
            trace
        })
        .collect()
}

fn bench_storage_links(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_links");
    let rt = Runtime::new().unwrap();

    for &count in &[100, 500, 1000] {
        let traces = generate_traces_with_links(count, 1536, 5);

        let mut storage = InMemoryStorage::new();
        rt.block_on(async {
            for trace in traces {
                storage.put(trace).await.unwrap();
            }
        });

        let link_count = count * std::cmp::min(5, count);
        group.throughput(Throughput::Elements(link_count as u64));
        group.bench_with_input(BenchmarkId::new("traces", count), &count, |bench, _| {
            bench.iter(|| {
                rt.block_on(async {
                    black_box(storage.links().await.unwrap());
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
        let traces = generate_traces(count * 2, 1536);
        let ids: Vec<Id> = traces.iter().map(|t| t.id).collect();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("operations", count), &count, |bench, _| {
            bench.iter(|| {
                let mut storage = InMemoryStorage::new();
                rt.block_on(async {
                    for i in 0..count {
                        storage.put(traces[i].clone()).await.unwrap();
                    }

                    for i in 0..count / 2 {
                        storage.get(ids[i]).await.unwrap();
                    }

                    for i in 0..count / 4 {
                        storage.remove(ids[i]).await.unwrap();
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
        let traces = generate_traces(count, 1536);
        let ids: Vec<Id> = traces.iter().map(|t| t.id).collect();

        let storage = Arc::new(Mutex::new(InMemoryStorage::new()));
        rt.block_on(async {
            let mut s = storage.lock().await;
            for trace in traces {
                s.put(trace).await.unwrap();
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
        let traces = generate_traces(count, 1536);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("concurrent", count), &count, |bench, _| {
            bench.iter(|| {
                let storage = Arc::new(Mutex::new(InMemoryStorage::new()));
                let traces = traces.clone();
                rt.block_on(async {
                    let mut s = storage.lock().await;
                    for trace in traces {
                        black_box(s.put(trace).await.unwrap());
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
        let traces = generate_traces(count, 1536);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("insert", count), &count, |bench, _| {
            bench.iter(|| {
                let mut storage = InMemoryStorage::new();
                rt.block_on(async {
                    for trace in traces.clone() {
                        black_box(storage.put(trace).await.unwrap());
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
        let traces = generate_traces(count, 1536);
        let ids: Vec<Id> = traces.iter().map(|t| t.id).collect();

        let mut storage = InMemoryStorage::new();
        rt.block_on(async {
            for trace in traces {
                storage.put(trace).await.unwrap();
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
        let traces = generate_traces(count, 1536);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("traces", count), &count, |bench, _| {
            bench.iter(|| {
                let storage = rt.block_on(async {
                    let s = SqliteStorage::new_in_memory().await.unwrap();
                    for trace in traces.clone() {
                        black_box(s.put(trace).await.unwrap());
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
        let traces = generate_traces(count, 1536);
        let ids: Vec<Id> = traces.iter().map(|t| t.id).collect();

        let storage = rt.block_on(async {
            let s = SqliteStorage::new_in_memory().await.unwrap();
            for trace in traces {
                s.put(trace).await.unwrap();
            }
            s
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("traces", count), &count, |bench, _| {
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
        let traces = generate_traces(count, 1536);

        let storage = rt.block_on(async {
            let s = SqliteStorage::new_in_memory().await.unwrap();
            for trace in traces {
                s.put(trace).await.unwrap();
            }
            s
        });

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("traces", count), &count, |bench, _| {
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
    let traces = generate_traces(count, 1536);
    let ids: Vec<Id> = traces.iter().map(|t| t.id).collect();

    group.throughput(Throughput::Elements(count as u64));

    group.bench_function("inmemory_put", |bench| {
        bench.iter(|| {
            let mut storage = InMemoryStorage::new();
            rt.block_on(async {
                for trace in traces.clone() {
                    black_box(storage.put(trace).await.unwrap());
                }
            });
            black_box(storage)
        });
    });

    group.bench_function("sqlite_put", |bench| {
        bench.iter(|| {
            let storage = rt.block_on(async {
                let s = SqliteStorage::new_in_memory().await.unwrap();
                for trace in traces.clone() {
                    black_box(s.put(trace).await.unwrap());
                }
                s
            });
            black_box(storage)
        });
    });

    let mut inmem_storage = InMemoryStorage::new();
    rt.block_on(async {
        for trace in traces.clone() {
            inmem_storage.put(trace).await.unwrap();
        }
    });

    let sqlite_storage = rt.block_on(async {
        let s = SqliteStorage::new_in_memory().await.unwrap();
        for trace in traces.clone() {
            s.put(trace).await.unwrap();
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
