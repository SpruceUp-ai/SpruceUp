import typing


def schema_hints(schema: type) -> dict[str, object]:
    return typing.get_type_hints(schema)


def is_embedding_type(tp) -> bool:
    return typing.get_origin(tp) is list and typing.get_args(tp) == (float,)


def validate_vector_column(schema: type, vector_column: str) -> None:
    hints = schema_hints(schema)
    if vector_column not in hints:
        raise ValueError(
            f"vector_column {vector_column!r} is not a field of "
            f"{schema.__name__!r}; available fields: {sorted(hints)}"
        )
    if not is_embedding_type(hints[vector_column]):
        raise ValueError(
            f"vector_column {vector_column!r} must be typed list[float], "
            f"got {hints[vector_column]!r}"
        )
