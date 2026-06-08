import asyncio
import importlib
import logging
import pathlib
import sys

from . import app
from .connectors.base import EmbeddingConfigError
from .utils.validation import validate_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "start":
        print("Usage: spruceup start [--no-cache-files]")
        sys.exit(1)

    cache_files = True
    for arg in sys.argv[2:]:
        if arg == "--no-cache-files":
            cache_files = False
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: spruceup start [--no-cache-files]")
            sys.exit(1)

    cwd = str(pathlib.Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        pipeline = importlib.import_module("spruceup_pipeline")
    except ModuleNotFoundError:
        sys.exit(
            """
                Error: spruceup_pipeline.py not found in the current directory.
                Run 'spruceup start' from the directory that contains your pipeline file.
            """
        )

    validate_pipeline(pipeline)
    try:
        asyncio.run(app.run(pipeline, cache_files=cache_files))
    except EmbeddingConfigError as exc:
        sys.exit(f"\nEmbedder configuration error: {exc}")
    except KeyboardInterrupt:
        print("\nSpruceUp manually aborted by user command")
