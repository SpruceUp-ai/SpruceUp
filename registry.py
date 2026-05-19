"""
SpruceUp pipeline registry.

Exposes two decorators the user applies in their spruceup_pipeline.py:

  @file_transform   — marks the function that parses a file into chunk strings
                      signature: (content: str, filename: str) -> list[str]

  @chunk_transform  — marks the function that builds schema objects from chunk strings
                      signature: (chunk_strs: list[str]) -> list[SomeChunk]

When spruceup_pipeline.py is imported, the decorators fire and register both
functions with the module-level tracker. main.py then calls
tracker.configure(db_path) before any database access.
"""

from typing import Callable

from monitoring.capture import TransformTracker

# Singleton tracker — _manifest_path is empty until main.py calls tracker.configure()
tracker: TransformTracker = TransformTracker("")

file_transform_fn: Callable | None = None
chunk_transform_fn: Callable | None = None


def file_transform(fn: Callable) -> Callable:
    """Decorator: register fn as the file-level transform (content → chunk strings)."""
    global file_transform_fn
    file_transform_fn = fn
    tracker.register(fn)
    return fn


def chunk_transform(fn: Callable) -> Callable:
    """Decorator: register fn as the chunk-level transform (chunk strings → schema objects)."""
    global chunk_transform_fn
    chunk_transform_fn = fn
    tracker.register(fn)
    return fn
