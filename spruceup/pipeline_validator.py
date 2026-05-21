import spruceup.registry as registry

_REQUIRED: list[tuple[str, type]] = [
    ("CHUNK_SCHEMA", type),
    ("TARGET_DB",    str),
    ("TARGET_TABLE", str),
    ("PRIMARY_KEY",  str),
    ("WATCHED_DIR",  str),
    ("PG_CONNSTR",   str),
]

_MISSING = object()


def validate_pipeline(pipeline) -> None:
    """Validate that all required constants are defined on the pipeline module.

    Raises SystemExit listing every problem found so the user can fix them all
    in one pass rather than discovering errors one at a time.
    """
    errors: list[str] = []

    for name, expected in _REQUIRED:
        val = getattr(pipeline, name, _MISSING)
        if val is _MISSING:
            errors.append(f"  {name} is not defined")
        elif expected is type and not isinstance(val, type):
            errors.append(f"  {name} must be a class, got {type(val).__name__!r}")
        elif expected is str and not isinstance(val, str):
            errors.append(f"  {name} must be a str, got {type(val).__name__!r}")

    if registry.transform_fn is None:
        errors.append("  no @transform function was registered")

    if errors:
        raise SystemExit(
            "spruceup_pipeline.py is misconfigured:\n" + "\n".join(errors)
        )
