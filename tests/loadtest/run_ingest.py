"""Instrumented cold-ingestion load test for SpruceUp.

Wires the real Coordinator / SyncEngine / Manifest against stub (or real)
connectors, enqueues one upsert per file in a corpus, drives the pipeline to
drain, and reports throughput, peak memory, peak in-flight concurrency, peak
queue depth, and error count.

Defaults isolate the pipeline (stub embedder + stub target, no network) so the
pipeline's own mechanics are what's measured. Use --target pg / --embed-latency-ms
to add back the real dependencies one at a time.

Key flags for the four benchmarks:

  (i)  Embedding batch vs. per-request overhead:
       Use bench_embed.py instead — it tests a real API directly.

  (ii) Embed time as fraction of total ingestion time:
       Run normally; the report always shows cumulative embed_batch time and
       what percentage of wall time it occupied.  Add --embed-latency-ms to
       simulate realistic API round-trips.

  (iii) File-content cache savings (run 2 reads raw_content from SQLite):
       python run_ingest.py --corpus /tmp/corpus --runs 2
       python run_ingest.py --corpus /tmp/corpus --runs 2 --no-cache-files
       Compare run-2 times: with vs. without --no-cache-files.

  (iv) Asyncio concurrency vs. sequential processing:
       python run_ingest.py --corpus /tmp/corpus --compare-sequential
       Runs the full ingest twice (concurrent then max_concurrency=1) and
       prints the speedup ratio.

Examples:
    python run_ingest.py --corpus /tmp/corpus_10k
    python run_ingest.py --corpus /tmp/corpus_10k --target pg --pg-table loadtest_chunks
    python run_ingest.py --corpus /tmp/corpus_10k --embed-latency-ms 50
    python run_ingest.py --corpus /tmp/corpus_10k --runs 2
    python run_ingest.py --corpus /tmp/corpus_10k --compare-sequential
"""

import argparse
import asyncio
import cProfile
import hashlib
import io
import logging
import os
import pathlib
import pstats
import resource
import time
import tracemalloc

import dotenv

from spruceup import LocalFilesSource, PgVectorTarget
from spruceup.connectors.embedders.embedding_batcher import EmbeddingBatcher
from spruceup.connectors.sources.local import make_file_id
from spruceup.coordinator import Coordinator
from spruceup.debounce_queue import DebounceQueue
from spruceup.manifest import Manifest
from spruceup.models import FileProps, SyncTask
from spruceup.sync_engine import SyncEngine

from tests.loadtest.stubs import LoadTestChunk, StubEmbedder, StubTarget


# --- transform (self-contained; splits on blank lines) ----------------

async def load_test_transform(*, file_props: FileProps, embed) -> list[LoadTestChunk]:
    raw = file_props.raw_content
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    name = file_props.display_name
    chunks = [p for p in text.split("\n\n") if p.strip()]
    if not chunks:
        return []
    embeddings = await embed(chunks)
    return [
        LoadTestChunk(
            id=hashlib.blake2b(f"{name}:{i}".encode(), digest_size=16).hexdigest(),
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
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    try:
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 if sys.platform == "linux" else rss / (1024 * 1024)
    except Exception:
        return 0.0


def _peak_rss_mb() -> float:
    try:
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 if sys.platform == "linux" else rss / (1024 * 1024)
    except Exception:
        return 0.0


# --- drain helper -----------------------------------------------------

async def _enqueue_and_drain(
    queue: DebounceQueue,
    coordinator: Coordinator,
    file_ids: list[str],
    data_source_id: int,
    timeout: float,
) -> tuple[float, bool]:
    """Enqueue all files then wait for the queue and coordinator to fully drain.

    file_ids must be in inode:path format as produced by make_file_id(), so that
    LocalFilesSource.fetch() can parse them back to a filesystem path.
    """
    t0 = time.monotonic()
    for fid in file_ids:
        await queue.put(SyncTask("upsert", fid, data_source_id))
    stable = 0
    deadline = t0 + timeout
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
    return time.monotonic() - t0, timed_out


# --- sequential comparison pass ---------------------------------------

async def _sequential_pass(
    args,
    source: LocalFilesSource,
    file_ids: list[str],
) -> tuple[float, bool]:
    """Run a fresh single-file-at-a-time ingest on a temporary manifest.

    Used by --compare-sequential to show how much asyncio concurrency contributes
    to throughput.  Everything is identical to the main run except max_concurrency=1.
    The temporary manifest is deleted on completion.
    """
    seq_path = args.manifest + ".seq"
    for suffix in ("", "-wal", "-shm"):
        p = seq_path + suffix
        if os.path.exists(p):
            os.remove(p)

    manifest_s = Manifest(seq_path)
    dsid_s = manifest_s.register_source(source.source_type, source.source_identifier)

    embedder_s = StubEmbedder(
        dimensions=args.embed_dims,
        latency_s=args.embed_latency_ms / 1000,
        max_batch_size=args.max_batch_size,
    )
    batcher_s = EmbeddingBatcher(embedder_s, max_batch_size=args.max_batch_size)
    target_s = StubTarget(latency_s=args.target_latency_ms / 1000)
    target_s.ensure_table_exists(embedder_s.embedding_dimensions)
    sync_engine_s = SyncEngine(manifest=manifest_s, target=target_s)
    queue_s: DebounceQueue = DebounceQueue()
    coordinator_s = Coordinator(
        queue=queue_s,
        transform=load_test_transform,
        embedder=batcher_s,
        sync_engine=sync_engine_s,
        manifest=manifest_s,
        target=target_s,
        source_registry={dsid_s: source},
        max_concurrency=1,
    )

    coord_task = asyncio.create_task(coordinator_s.run())
    elapsed, timed_out = await _enqueue_and_drain(queue_s, coordinator_s, file_ids, dsid_s, args.timeout)
    coord_task.cancel()
    await asyncio.gather(coord_task, return_exceptions=True)
    manifest_s.close()
    for suffix in ("", "-wal", "-shm"):
        p = seq_path + suffix
        if os.path.exists(p):
            os.remove(p)

    return elapsed, timed_out


# --- driver -----------------------------------------------------------

async def drive(args) -> None:
    if os.path.exists(args.manifest):
        os.remove(args.manifest)
    manifest = Manifest(args.manifest)

    source = LocalFilesSource(watched_dir=args.corpus)
    data_source_id = manifest.register_source(source.source_type, source.source_identifier)
    source_registry = {data_source_id: source}

    if args.index_chunks:
        manifest._conn.execute("CREATE INDEX IF NOT EXISTS ix_chunks_file_id ON chunks(file_id)")

    if args.shared_conn:
        # Diagnostic: Manifest already uses a single persistent connection opened at
        # construction time (self._conn), so this flag is now a no-op — the
        # "always-shared-conn" behaviour is the current default.
        pass

    if args.skip_manifest:
        # Diagnostic: replace all SQLite access with no-ops to isolate whether
        # the manifest's SQLite usage is the source of any measured overhead.
        manifest.get_chunks_for_file = lambda *a, **k: []
        manifest.get_file_modified_at = lambda *a, **k: None
        manifest.get_cached_embeddings = lambda *a, **k: {}
        manifest.set_cached_embeddings = lambda *a, **k: None
        manifest.upsert_file_row = lambda *a, **k: None
        manifest.upsert_chunks = lambda *a, **k: None
        manifest.sweep_memoized = lambda *a, **k: None
        manifest.sweep_embedding_cache = lambda *a, **k: None
        manifest.set_sync_state = lambda *a, **k: None
        manifest.mark_failed = lambda *a, **k: None
        manifest.mark_fetch_failed = lambda *a, **k: None
        manifest.delete_chunks = lambda *a, **k: None
        manifest.delete_file_row = lambda *a, **k: None

    if args.target == "pg":
        dotenv.load_dotenv()
        target = PgVectorTarget(
            connstr=os.getenv("PG_CONNSTR"),
            table=args.pg_table,
            schema=LoadTestChunk,
            vector_column="chunk_embedding",
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
    queue: DebounceQueue = DebounceQueue()
    coordinator = Coordinator(
        queue=queue,
        transform=load_test_transform,
        embedder=batcher,
        sync_engine=sync_engine,
        manifest=manifest,
        target=target,
        source_registry=source_registry,
        max_concurrency=args.max_concurrency,
        cache_files=not args.no_cache_files,
    )

    errors = _ErrorCounter()
    logging.getLogger("spruceup").addHandler(errors)

    files = sorted(str(p) for p in pathlib.Path(args.corpus).rglob("*") if p.is_file())
    n_files = len(files)
    # LocalFilesSource.fetch() parses file_id as inode:path via file_id_to_path()
    file_ids = [make_file_id(os.stat(fp).st_ino, fp) for fp in files]

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

    profiler = cProfile.Profile() if args.profile else None

    rss_start = _current_rss_mb()
    coordinator_task = asyncio.create_task(coordinator.run())
    sampler_task = asyncio.create_task(sampler())
    trim_task = asyncio.create_task(trimmer()) if args.trim_interval_ms > 0 else None

    # --- main ingest runs ------------------------------------------------
    # Each run re-enqueues all files on the same manifest/coordinator.  Run 2+
    # exercises the file-content cache (raw_content stored in SQLite) and the
    # embedding cache (chunk embeddings stored in SQLite), so run-2 time vs.
    # run-1 time shows the combined cache speedup.

    run_records: list[dict] = []
    for run_idx in range(args.runs):
        embed_t0 = embedder.total_embed_time_s
        upserts_t0 = target.upserts if args.target != "pg" else 0

        if profiler:
            profiler.enable()
        elapsed, timed_out = await _enqueue_and_drain(queue, coordinator, file_ids, data_source_id, args.timeout)
        if profiler:
            profiler.disable()

        run_records.append({
            "run": run_idx + 1,
            "elapsed": elapsed,
            "timed_out": timed_out,
            "embed_time_s": embedder.total_embed_time_s - embed_t0,
            "upserts": (target.upserts - upserts_t0) if args.target != "pg" else None,
        })

    # --- catchup mode (only after the last run) --------------------------
    catchup_elapsed = None
    if args.mode == "catchup":
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
        *([trim_task] if trim_task is not None else []),
        return_exceptions=True,
    )

    # --- sequential comparison pass (fresh manifest, max_concurrency=1) --
    seq_elapsed: float | None = None
    seq_timed_out = False
    if args.compare_sequential:
        print("\nRunning sequential comparison pass (max_concurrency=1) …")
        seq_elapsed, seq_timed_out = await _sequential_pass(args, source, file_ids)

    # --- collect final metrics -------------------------------------------
    if args.target == "pg":
        import psycopg
        with psycopg.connect(os.getenv("PG_CONNSTR")) as conn:
            chunks_written = conn.execute(f"SELECT count(*) FROM {args.pg_table}").fetchone()[0]
    else:
        chunks_written = run_records[0]["upserts"] or 0

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

    elapsed = run_records[0]["elapsed"]
    timed_out = run_records[0]["timed_out"]
    total_wall = sum(r["elapsed"] for r in run_records)
    total_embed = embedder.total_embed_time_s

    # --- report ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  corpus:            {args.corpus}")
    print(f"  target / embedder: {args.target} / stub (latency {args.embed_latency_ms} ms)")
    print(f"  max concurrency:   {args.max_concurrency}")
    print(f"  files:             {n_files}")
    print(f"  chunks written:    {chunks_written}  (run 1 net upserts to target)")

    if args.runs == 1:
        flag = "  [TIMED OUT]" if timed_out else ""
        print(f"  wall time:         {elapsed:.2f} s{flag}")
        if catchup_elapsed is not None:
            print(f"  catch-up scan:     {catchup_elapsed:.2f} s   <- no-op rescan of {n_files} unchanged files")
        if elapsed > 0:
            print(f"  files/sec:         {n_files / elapsed:.1f}")
            print(f"  chunks/sec:        {chunks_written / elapsed:.1f}")
    else:
        # Per-run breakdown
        for rec in run_records:
            label = "cold" if rec["run"] == 1 else "warm"
            flag = "  [TIMED OUT]" if rec["timed_out"] else ""
            fps = f"  ({n_files / rec['elapsed']:.0f} files/s)" if rec["elapsed"] > 0 else ""
            embed_pct = (rec["embed_time_s"] / rec["elapsed"] * 100) if rec["elapsed"] > 0 else 0
            print(
                f"  run {rec['run']} ({label}):         {rec['elapsed']:.2f} s{flag}{fps}"
                f"  [embed {rec['embed_time_s']:.2f} s cumulative = {embed_pct:.0f}%]"
            )
        if len(run_records) >= 2:
            r1, r2 = run_records[0]["elapsed"], run_records[1]["elapsed"]
            speedup = r1 / r2 if r2 > 0 else float("inf")
            note = "file-content + embed cache warm on run 2"
            if args.no_cache_files:
                note = "embed cache warm on run 2 (file-content cache disabled)"
            print(f"  cache speedup:     {speedup:.2f}x   <- {note}")
        if catchup_elapsed is not None:
            print(f"  catch-up scan:     {catchup_elapsed:.2f} s   <- no-op rescan of {n_files} unchanged files")

    if args.runs == 1 and args.embed_latency_ms > 0:
        embed_pct = (total_embed / elapsed * 100) if elapsed > 0 else 0
        print(
            f"  embed API time:    {total_embed:.2f} s cumulative  ({embed_pct:.0f}% of wall)"
            f"   <- sum of all embed_batch call durations (concurrent calls overlap)"
        )

    if seq_elapsed is not None:
        flag = "  [TIMED OUT]" if seq_timed_out else ""
        print(f"\n  sequential pass:   {seq_elapsed:.2f} s{flag}  (max_concurrency=1)")
        if seq_elapsed > 0 and elapsed > 0:
            ratio = seq_elapsed / elapsed
            print(f"  concurrency gain:  {ratio:.1f}x faster with max_concurrency={args.max_concurrency}")

    print(f"\n  peak in-flight:    {stats['max_active']}   <- should be capped at max concurrency")
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
    if profiler:
        buf = io.StringIO()
        ps = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        ps.print_stats(20)
        print("\n  -- cProfile top 20 by cumulative time --")
        for line in buf.getvalue().splitlines()[4:]:
            print(" ", line)

    corpus_size_mb = sum(os.path.getsize(fp) for fp in files) / 1e6
    manifest_size_mb = os.path.getsize(args.manifest) / 1e6
    print(f"\n  corpus size:       {corpus_size_mb:.1f} MB  ({n_files} files × {corpus_size_mb / n_files * 1024:.0f} KB avg)")
    print(f"  manifest size:     {manifest_size_mb:.1f} MB  ({manifest_size_mb / corpus_size_mb:.2f}× corpus)")
    print(f"  errors logged:     {errors.count}")
    for s in errors.samples:
        print(f"      e.g. {s[:90]}")
    print("=" * 60)

    await batcher.aclose()
    manifest.close()


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
    ap.add_argument("--runs", type=int, default=1, metavar="N",
                    help="Run ingest N times on the same manifest without resetting between runs. "
                         "Run 2+ use the warm file-content and embedding caches. (default: 1)")
    ap.add_argument("--no-cache-files", action="store_true",
                    help="Disable caching raw file content in the manifest (coordinator cache_files=False). "
                         "Use with --runs 2 to isolate the file-content cache contribution.")
    ap.add_argument("--compare-sequential", action="store_true",
                    help="After the main run, perform a fresh ingest with max_concurrency=1 and "
                         "report the speedup ratio from asyncio concurrency.")
    ap.add_argument("--trim-interval-ms", type=float, default=0.0, help="periodic malloc_trim(0); 0 = off")
    ap.add_argument("--skip-manifest", action="store_true", help="diagnostic: no-op all SQLite manifest access")
    ap.add_argument("--index-chunks", action="store_true", help="diagnostic: add an index on chunks(file_id)")
    ap.add_argument("--shared-conn", action="store_true", help="diagnostic: no-op (manifest already uses a single persistent connection)")
    ap.add_argument("--profile", action="store_true", help="diagnostic: profile ingest runs with cProfile and print top 20 functions by cumulative time")
    ap.add_argument("--mode", choices=["enqueue", "catchup"], default="enqueue",
                    help="catchup also times a no-op catch-up scan on the populated manifest")
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
