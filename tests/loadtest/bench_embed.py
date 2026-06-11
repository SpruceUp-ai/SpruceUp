"""Micro-benchmark: batched vs. parallel-single vs. serial-single embedding calls.

Measures the per-request overhead of an embedding API by comparing three strategies
for embedding the same N chunks:

  A. Batched       — 1 API request containing all N chunks
  B. Parallel      — N concurrent API requests each containing 1 chunk
  C. Serial        — N sequential API requests each containing 1 chunk

The gap between A and B shows how much of the parallel overhead is pure round-trip
cost (connection setup, TLS, HTTP framing) that batching eliminates. The gap between
B and C shows how much asyncio concurrency helps when you can't batch.

Example:
    python bench_embed.py --provider openai --n-chunks 10 --repeats 5
    python bench_embed.py --provider openai --n-chunks 10 --repeats 5 --model text-embedding-3-large
    python bench_embed.py --provider openai --n-chunks 50 --repeats 3 --chunk-words 100
"""

import argparse
import asyncio
import random
import statistics
import time

_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris"
).split()


def _make_chunks(n: int, words: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(_WORDS) for _ in range(words)) for _ in range(n)]


async def _time_batched(embedder, chunks: list[str]) -> float:
    t0 = time.monotonic()
    await embedder.embed_batch(chunks)
    return time.monotonic() - t0


async def _time_parallel(embedder, chunks: list[str]) -> float:
    t0 = time.monotonic()
    await asyncio.gather(*[embedder.embed_batch([c]) for c in chunks])
    return time.monotonic() - t0


async def _time_serial(embedder, chunks: list[str]) -> float:
    t0 = time.monotonic()
    for c in chunks:
        await embedder.embed_batch([c])
    return time.monotonic() - t0


def _fmt(times: list[float]) -> str:
    mean_ms = statistics.mean(times) * 1000
    if len(times) == 1:
        return f"{mean_ms:.0f} ms"
    return f"{mean_ms:.0f} ms  (±{statistics.stdev(times) * 1000:.0f} ms  n={len(times)})"


async def run(args) -> None:
    if args.provider == "openai":
        import os
        import dotenv
        dotenv.load_dotenv()
        from spruceup.connectors.embedders.openai import OpenAIEmbedder
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("--api-key or OPENAI_API_KEY env var is required")
        embedder = OpenAIEmbedder(api_key=api_key, model=args.model)
    else:
        raise SystemExit(f"unknown provider: {args.provider!r}")

    chunks = _make_chunks(args.n_chunks, args.chunk_words)

    print(f"\nProvider / model: {args.provider} / {args.model}")
    print(f"Chunks:           {args.n_chunks}  ({args.chunk_words} words each)")
    print(f"Repeats:          {args.repeats}  (+ 1 warm-up)\n")

    # warm-up to establish the HTTP connection pool before timing
    await embedder.embed_batch(chunks[:1])

    batched_times: list[float] = []
    parallel_times: list[float] = []
    serial_times: list[float] = []

    for rep in range(args.repeats):
        b = await _time_batched(embedder, chunks)
        p = await _time_parallel(embedder, chunks)
        s = await _time_serial(embedder, chunks)
        batched_times.append(b)
        parallel_times.append(p)
        serial_times.append(s)
        print(
            f"  rep {rep + 1}/{args.repeats}"
            f"  batched={b * 1000:.0f}ms"
            f"  parallel={p * 1000:.0f}ms"
            f"  serial={s * 1000:.0f}ms"
        )

    mean_b = statistics.mean(batched_times)
    mean_p = statistics.mean(parallel_times)
    mean_s = statistics.mean(serial_times)

    # Per-request overhead: parallel runs N requests concurrently; the extra
    # wall time vs. batched is the cost of N-1 extra round-trips firing in
    # parallel. Divide by N to get per-request overhead estimate.
    overhead_total_ms = (mean_p - mean_b) * 1000
    overhead_per_req_ms = overhead_total_ms / args.n_chunks

    print("\n" + "=" * 64)
    print(f"  batched  (1 req, {args.n_chunks} chunks):          {_fmt(batched_times)}")
    print(f"  parallel ({args.n_chunks} reqs, concurrent):       {_fmt(parallel_times)}")
    print(f"  serial   ({args.n_chunks} reqs, sequential):       {_fmt(serial_times)}")
    print()
    print(f"  parallel / batched:  {mean_p / mean_b:.2f}x slower")
    print(f"  serial   / batched:  {mean_s / mean_b:.2f}x slower")
    print(f"  serial   / parallel: {mean_s / mean_p:.2f}x slower  (asyncio concurrency gain)")
    print()
    print(
        f"  per-request round-trip overhead: ~{overhead_per_req_ms:.1f} ms"
        f"  (extra {overhead_total_ms:.0f} ms for {args.n_chunks} parallel requests vs 1 batch)"
    )
    print("=" * 64 + "\n")

    await embedder.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare batched vs. parallel-single vs. serial-single embedding latency."
    )
    ap.add_argument("--provider", default="openai", choices=["openai"],
                    help="Embedding provider (default: openai)")
    ap.add_argument("--model", default="text-embedding-3-small",
                    help="Model name (default: text-embedding-3-small)")
    ap.add_argument("--api-key", default=None,
                    help="API key (falls back to OPENAI_API_KEY env var)")
    ap.add_argument("--n-chunks", type=int, default=10,
                    help="Number of chunks to embed per trial (default: 10)")
    ap.add_argument("--chunk-words", type=int, default=50,
                    help="Words per chunk (default: 50)")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Number of timed repetitions per strategy (default: 5)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
