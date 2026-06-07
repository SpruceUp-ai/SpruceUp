import json

SUPPORTED_TYPES = (str, int, float, bool, list, dict)


def validate_return_type(returns) -> None:
    if returns not in SUPPORTED_TYPES:
        raise TypeError(
            f"@memoize(returns={returns!r}) is not supported. "
            f"Supported: str, int, float, bool, list, dict"
        )


def serialize(value, returns) -> bytes:
    if returns is dict and not all(isinstance(k, str) for k in value):
        raise TypeError(
            "@memoize: dict return values must have string keys for JSON serialization"
        )
    return json.dumps(value).encode()


def deserialize(data: bytes, returns):
    value = json.loads(data)
    if returns is float and isinstance(value, int):
        return float(value)
    return value
