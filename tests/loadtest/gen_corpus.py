"""Generate a synthetic text corpus for load testing.

Files are plain text with `chunks-per-file` paragraphs separated by blank
lines, which the load-test transform splits on. This gives precise control
over file count, file size, and total chunk count — the three axes we ramp.

Examples:
    # 10k small files, 10 chunks each  -> 100k chunks
    python gen_corpus.py --dir /tmp/corpus_10k --count 10000 --chunks-per-file 10

    # one very large file -> memory-axis test
    python gen_corpus.py --dir /tmp/corpus_big --count 1 --chunks-per-file 200000
"""

import argparse
import pathlib
import random
import shutil

_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo "
    "consequat duis aute irure reprehenderit voluptate velit esse cillum"
).split()


def _paragraph(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--count", type=int, default=1000, help="number of files")
    ap.add_argument("--chunks-per-file", type=int, default=10)
    ap.add_argument("--chunk-words", type=int, default=60)
    ap.add_argument("--ext", default=".txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="wipe dir first")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total_bytes = 0
    for i in range(args.count):
        paragraphs = [
            _paragraph(rng, args.chunk_words) for _ in range(args.chunks_per_file)
        ]
        text = "\n\n".join(paragraphs) + "\n"
        path = root / f"file_{i:07d}{args.ext}"
        path.write_text(text, encoding="utf-8")
        total_bytes += len(text)

    print(
        f"Generated {args.count} file(s) in {root}  "
        f"chunks/file={args.chunks_per_file}  "
        f"total_chunks={args.count * args.chunks_per_file}  "
        f"total_size={total_bytes / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
