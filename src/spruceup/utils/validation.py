from spruceup.config import SpruceUpConfig


def validate_schema_objects(objs: list, schema_class: type) -> None:
    if not isinstance(objs, list):
        raise ValueError(
            f"transform must return a list, got {type(objs).__name__!r}"
        )
    for i, obj in enumerate(objs):
        if not isinstance(obj, schema_class):
            raise ValueError(
                f"transform returned {type(obj).__name__!r} at index {i}, "
                f"expected {schema_class.__name__!r}"
            )

def validate_pipeline(pipeline) -> None:
    errors: list[str] = []

    config = getattr(pipeline, "config", None)
    if config is None:
        errors.append("  config is not defined — call config = defineConfig(...)")
    elif not isinstance(config, SpruceUpConfig):
        errors.append(
            f"  config must be the result of defineConfig(), got {type(config).__name__!r}"
        )

    if config is not None and isinstance(config, SpruceUpConfig) and config.transform is None:
        errors.append("  no transform function was provided to defineConfig()")

    if errors:
        raise SystemExit(
            "spruceup_pipeline.py is misconfigured:\n" + "\n".join(errors)
        )
