import spruceup.registry as registry
from spruceup.config import SpruceUpConfig


def validate_pipeline(pipeline) -> None:
    """Validate that the pipeline module has a valid config and a registered @transform.

    Raises SystemExit listing every problem found so the user can fix them all
    in one pass rather than discovering errors one at a time.
    """
    errors: list[str] = []

    config = getattr(pipeline, "config", None)
    if config is None:
        errors.append("  config is not defined — call config = defineConfig(...)")
    elif not isinstance(config, SpruceUpConfig):
        errors.append(
            f"  config must be the result of defineConfig(), got {type(config).__name__!r}"
        )

    if registry.transform_fn is None:
        errors.append("  no @transform function was registered")

    if errors:
        raise SystemExit(
            "spruceup_pipeline.py is misconfigured:\n" + "\n".join(errors)
        )
