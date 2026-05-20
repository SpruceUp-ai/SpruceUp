"""
SpruceUp pipeline registry.

Exposes one decorator the user applies in their spruceup_pipeline.py:

  @transform  — marks the single async function that converts a file into
                schema objects ready for storage.
                signature: (*, file_props: dict, embed) -> list[SomeChunk]

When spruceup_pipeline.py is imported, the decorator fires and registers the
function with the module-level tracker. main.py then calls
tracker.configure(manifest_path) before any database access.
"""

from typing import Callable

from .monitoring.capture import TransformTracker

# Singleton tracker — _manifest_path is empty until main.py calls tracker.configure()
tracker: TransformTracker = TransformTracker("")

transform_fn: Callable | None = None


def transform(fn: Callable) -> Callable:
    """Decorator: register fn as the pipeline transform (file_props + embed → schema objects)."""
    global transform_fn
    transform_fn = fn
    tracker.register(fn)
    return fn
