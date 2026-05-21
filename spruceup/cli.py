import asyncio
import importlib
import logging
import pathlib
import sys

from . import app
from .pipeline_validator import validate_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "start":
        print("Usage: spruceup start")
        sys.exit(1)

    # Ensure the user's CWD is importable so their pipeline file and any
    # local imports inside it (e.g. from example.dummy_pipeline import ...) resolve.
    cwd = str(pathlib.Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        pipeline = importlib.import_module("spruceup_pipeline")
    except ModuleNotFoundError:
        sys.exit(
            "Error: spruceup_pipeline.py not found in the current directory.\n"
            "Run 'spruceup start' from the directory that contains your pipeline file."
        )

    validate_pipeline(pipeline)
    try:
        asyncio.run(app.run(pipeline))
    except KeyboardInterrupt:
        print("\nSpruceUp manually aborted by user command")
