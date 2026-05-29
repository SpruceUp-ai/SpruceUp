"""Instrumented cold-ingestion load test for SpruceUp.

Wires the real Coordinator / SyncEngine / Manifest against stub (or real)
connectors, enqueues one upsert per file in a corpus, drives the pipeline to
drain, and reports throughput, peak memory, peak in-flight concurrency, peak
queue depth, and error count.

Defaults isolate the pipeline (stub embedder + stub target, no network) so the
pipeline's own mechanics are what's measured. Use --target pg / --embed-latency-ms
to add back the real dependencies one at a time.

Example:
    python run_ingest.py --corpus /tmp/corpus_10k
    python run_ingest.py --corpus /tmp/corpus_10k --target pg --pg-table loadtest_chunks
    python run_ingest.py --corpus /tmp/corpus_10k --embed-latency-ms 50
"""

import argparse
import asyncio
import hashlib
import logging
import os
import pathlib
import resource
import time
import tracemalloc

import dotenv

from spruceup import LocalFilesSource, PgVectorTarget
from spruceup.connectors.embedders.embedding_batcher import EmbeddingBatcher
from spruceup.coordinator import Coordinator
from spruceup.manifest import Manifest
from spruceup.models import SyncTask
from spruceup.sync_engine import SyncEngine

from stubs import LoadTestChunk, StubEmbedder, StubTarget


# --- transform (self-contained; splits on blank lines) ----------------

async def load_test_transform(*, file_props: dict, embed) -> list[LoadTestChunk]:
    text = file_props["raw_content"]
    path = file_props["file_path"]
    name = pathlib.Path(path).name
    chunks = [p for p in text.split("\n\n") if p.strip()]
    if not chunks:
        return []
    embeddings = await embed(chunks)
    return [
        LoadTestChunk(
            id=hashlib.blake2b(f"{path}:{i}".encode(), digest_size=16).hexdigest(),
            chunk_text=c,
            chunk_embedding=e,
            source_file=name,
        )
        for i, (c, e) in enumerate(zip(chunks, embeddings))
    ]


# --- instrumentation helpers ------------------------------------------

class _ErrorCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.count = 0
        self.samples: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1
        if len(self.samples) < 5:
            self.samples.append(record.getMessage())


def _current_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# --- driver -----------------------------------------------------------

async def drive(args) -> None:
    if os.path.exists(args.manifest):
        os.remove(args.manifest)
    manifest = Manifest(args.manifest)

    source = LocalFilesSource(watched_dir=args.corpus)
    data_source_id = manifest.register_source(source.source_type, source.source_identifier)
    source_registry = {data_source_id: source}

    if args.index_chunks:
        with manifest.connect() as con:
            con.execute("CREATE INDEX IF NOT EXISTS ix_chunks_file_id ON chunks(file_id)")

    if args.index_files:
        with manifest.connect() as con:
            con.execute("CREATE INDEX IF NOT EXISTS ix_files_inode_src ON files(inode, data_source_id)")
            con.execute("CREATE INDEX IF NOT EXISTS ix_files_src ON files(data_source_id)")

    if args.shared_conn:
        # Diagnostic: reuse ONE real SQLite connection for the whole run instead
        # of opening/closing a fresh one per manifest call.
        import sqlite3

        class _SharedConnProxy:
            def __init__(self, real): self._real = real
            def __enter__(self): self._real.__enter__(); return self
            def __exit__(self, *a): return self._real.__exit__(*a)
            def execute(self, *a, **k): return self._real.execute(*a, **k)
            def executemany(self, *a, **k): return self._real.executemany(*a, **k)
            def cursor(self): return self._real.cursor()
            def commit(self): return self._real.commit()
            def close(self): pass

        _real = sqlite3.connect(args.manifest)
        _real.execute("PRAGMA foreign_keys = ON")
        manifest.connect = lambda: _SharedConnProxy(_real)

    if args.skip_manifest:
        # Diagnostic: replace all SQLite access with no-ops to isolate whether
        # the manifest's SQLite usage is the source of the C-level memory growth.
        class _FakeConn:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a, **k): return self
            def executemany(self, *a, **k): return self
            def cursor(self): return self
            def fetchone(self): return None
            def fetchall(self): return []
            def __iter__(self): return iter(())
            def commit(self): pass
            def close(self): pass
        manifest.connect = lambda: _FakeConn()
        manifest.get_chunks_for_file = lambda *a, **k: []
        manifest.upsert_chunks = lambda *a, **k: None
        manifest.ensure_file_row_exists = lambda *a, **k: None
        manifest.upsert_file_row = lambda *a, **k: None
        manifest.sweep_memoized = lambda *a, **k: None
        manifest.delete_chunks = lambda *a, **k: None
        manifest.delete_file_row = lambda *a, **k: None

    if args.target == "pg":
        dotenv.load_dotenv()
        target = PgVectorTarget(
            connstr=os.getenv("PG_CONNSTR"),
            table=args.pg_table,
            schema=LoadTestChunk,
            primary_key="id",
        )
    else:
        target = StubTarget(latency_s=args.target_latency_ms / 1000)

    embedder = StubEmbedder(
        dimensions=args.embed_dims,
        latency_s=args.embed_latency_ms / 1000,
        max_batch_size=args.max_batch_size,
    )
    batcher = EmbeddingBatcher(embedder, max_batch_size=args.max_batch_size)

    target.ensure_table_exists(embedder.embedding_dimensions)

    sync_engine = SyncEngine(manifest=manifest, target=target)
    queue: asyncio.Queue = asyncio.Queue()
    coordinator = Coordinator(
        queue=queue,
        transform=load_test_transform,
        embedder=batcher,
        sync_engine=sync_engine,
        source_registry=source_registry,
        max_concurrency=args.max_concurrency,
    )

    errors = _ErrorCounter()
    logging.getLogger("spruceup").addHandler(errors)

    files = sorted(str(p) for p in pathlib.Path(args.corpus).rglob("*") if p.is_file())
    n_files = len(files)

    if args.trace:
        tracemalloc.start(15)

    stats = {"max_active": 0, "max_qdepth": 0, "peak_snapshot": None, "snapped": False}
    stop_sampler = asyncio.Event()

    async def sampler() -> None:
        while not stop_sampler.is_set():
            active = len(coordinator._active_tasks)
            stats["max_active"] = max(stats["max_active"], active)
            stats["max_qdepth"] = max(stats["max_qdepth"], queue.qsize())
            if args.trace and not stats["snapped"] and active >= 0.8 * n_files and n_files > 1:
                stats["peak_snapshot"] = tracemalloc.take_snapshot()
                stats["snapped"] = True
            await asyncio.sleep(0.005)

    async def trimmer() -> None:
        import ctypes
        malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
        while not stop_sampler.is_set():
            await asyncio.sleep(args.trim_interval_ms / 1000)
            malloc_trim(0)

    rss_start = _current_rss_mb()
    coordinator_task = asyncio.create_task(coordinator.run())
    sampler_task = asyncio.create_task(sampler())
    trim_task = asyncio.create_task(trimmer()) if args.trim_interval_ms > 0 else None

    t0 = time.monotonic()
    for fp in files:
        await queue.put(SyncTask(source.source_type, fp, "upsert", data_source_id=data_source_id))

    stable = 0
    deadline = t0 + args.timeout
    timed_out = False
    while True:
        await asyncio.sleep(0.02)
        if queue.qsize() == 0 and len(coordinator._active_tasks) == 0:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        if time.monotonic() > deadline:
            timed_out = True
            break
    elapsed = time.monotonic() - t0

    catchup_elapsed = None
    if args.mode == "catchup":
        # The manifest is now fully populated. Re-run the real catch-up scan on
        # the unchanged corpus to measure the incremental no-op cost: this is the
        # files(inode, data_source_id) lookup path (one SELECT per file).
        watcher = source.create_watcher(data_source_id)
        cu0 = time.monotonic()
        await watcher._catch_up(queue, manifest, force_reindex=False)
        stable = 0
        while True:
            await asyncio.sleep(0.02)
            if queue.qsize() == 0 and len(coordinator._active_tasks) == 0:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
        catchup_elapsed = time.monotonic() - cu0

    stop_sampler.set()
    coordinator_task.cancel()
    if trim_task is not None:
        trim_task.cancel()
    await asyncio.gather(
        sampler_task,
        coordinator_task,
        *( [trim_task] if trim_task is not None else [] ),
        return_exceptions=True,
    )

    if args.target == "pg":
        import psycopg
        with psycopg.connect(os.getenv("PG_CONNSTR")) as conn:
            chunks_written = conn.execute(f"SELECT count(*) FROM {args.pg_table}").fetchone()[0]
    else:
        chunks_written = target.upserts

    peak = _peak_rss_mb()
    rss_end = _current_rss_mb()
    import gc
    gc.collect()
    rss_gc = _current_rss_mb()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    rss_trim = _current_rss_mb()

    print("\n" + "=" * 60)
    print(f"  corpus:            {args.corpus}")
    print(f"  target / embedder: {args.target} / stub (latency {args.embed_latency_ms} ms)")
    print(f"  max concurrency:   {args.max_concurrency}")
    print(f"  files:             {n_files}")
    print(f"  chunks written:    {chunks_written}")
    print(f"  wall time:         {elapsed:.2f} s" + ("  [TIMED OUT]" if timed_out else ""))
    if catchup_elapsed is not None:
        print(f"  catch-up scan:     {catchup_elapsed:.2f} s   <- no-op rescan of {n_files} unchanged files")
    if elapsed > 0:
        print(f"  files/sec:         {n_files / elapsed:.1f}")
        print(f"  chunks/sec:        {chunks_written / elapsed:.1f}")
    print(f"  peak in-flight:    {stats['max_active']}   <- should be capped at max concurrency")
    print(f"  peak queue depth:  {stats['max_qdepth']}")
    print(f"  RSS start -> peak: {rss_start:.0f} -> {peak:.0f} MB")
    print(f"  RSS end/gc/trim:   {rss_end:.0f} / {rss_gc:.0f} / {rss_trim:.0f} MB   <- end vs peak = transient? trim = glibc-reclaimable")
    if args.trace:
        _, tm_peak = tracemalloc.get_traced_memory()
        print(f"  live objs (peak):  {tm_peak / 1e6:.0f} MB   <- vs RSS shows allocator retention")
        snap = stats["peak_snapshot"]
        if snap is not None:
            print("  top live allocations near peak concurrency:")
            for stat in snap.statistics("lineno")[:8]:
                fr = stat.traceback[0]
                loc = f"{pathlib.Path(fr.filename).name}:{fr.lineno}"
                print(f"      {stat.size / 1e6:6.1f} MB  {loc}")
        tracemalloc.stop()
    print(f"  errors logged:     {errors.count}")
    for s in errors.samples:
        print(f"      e.g. {s[:90]}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--target", choices=["stub", "pg"], default="stub")
    ap.add_argument("--pg-table", default="loadtest_chunks")
    ap.add_argument("--embed-dims", type=int, default=1536)
    ap.add_argument("--embed-latency-ms", type=float, default=0.0)
    ap.add_argument("--target-latency-ms", type=float, default=0.0)
    ap.add_argument("--max-batch-size", type=int, default=150)
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--trim-interval-ms", type=float, default=0.0, help="periodic malloc_trim(0); 0 = off")
    ap.add_argument("--skip-manifest", action="store_true", help="diagnostic: no-op all SQLite manifest access")
    ap.add_argument("--index-chunks", action="store_true", help="diagnostic: add an index on chunks(file_id)")
    ap.add_argument("--shared-conn", action="store_true", help="diagnostic: reuse one SQLite connection for the whole run")
    ap.add_argument("--index-files", action="store_true", help="diagnostic: add indexes on files(inode, data_source_id)")
    ap.add_argument("--mode", choices=["enqueue", "catchup"], default="enqueue", help="catchup also times a no-op catch-up scan on the populated manifest")
    ap.add_argument("--manifest", default="/tmp/loadtest_manifest.db")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--quiet", action="store_true", help="suppress pipeline INFO logs")
    ap.add_argument("--trace", action="store_true", help="track live-object memory via tracemalloc")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(drive(args))


if __name__ == "__main__":
    main()
