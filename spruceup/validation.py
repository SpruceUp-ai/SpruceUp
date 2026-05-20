_MISSING = object()


def validate_schema_objects(objs: list, schema_class: type, primary_key: str) -> None:
    """Raise ValueError if objs is not a valid list of schema_class instances.

    Called after the user's @transform function returns, before chunk wrappers
    are built. Gives a clear error message when the user forgets to type their
    transform or accidentally returns the wrong type.
    """
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
        pk_val = getattr(obj, primary_key, _MISSING)
        if pk_val is _MISSING:
            raise ValueError(
                f"object at index {i} has no field {primary_key!r} "
                f"(declared as PRIMARY_KEY in pipeline)"
            )
        if pk_val is None:
            raise ValueError(
                f"object at index {i} has {primary_key!r} = None"
            )
