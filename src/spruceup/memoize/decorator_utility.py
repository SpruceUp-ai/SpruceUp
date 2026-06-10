import json

SUPPORTED_TYPES = (str, int, float, bool, list, dict)


def validate_return_type(memoized_subfn_return_type) -> None:
    if memoized_subfn_return_type not in SUPPORTED_TYPES:
        raise TypeError(
            f"@memoize(memoized_subfn_return_type={memoized_subfn_return_type!r}) is not supported. "
            f"Supported: str, int, float, bool, list, dict"
        )


def serialize(value, memoized_subfn_return_type) -> bytes:
    if memoized_subfn_return_type is dict and not all(isinstance(k, str) for k in value):
        raise TypeError(
            "@memoize: dict return values must have string keys for JSON serialization"
        )
    return json.dumps(value).encode()


def deserialize(data: bytes, memoized_subfn_return_type):
    value = json.loads(data)
    if memoized_subfn_return_type is float and isinstance(value, int):
        return float(value)
    return value
