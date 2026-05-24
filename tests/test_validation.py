"""Tests for validate_schema_objects."""

import pytest
from dataclasses import dataclass

from spruceup.utils.validation import validate_schema_objects


@dataclass
class GoodChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]


@dataclass
class OtherChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]


def good(pk="pk1") -> GoodChunk:
    return GoodChunk(id=pk, chunk_text="text", chunk_embedding=[0.1])


class TestValidateSchemaObjects:

    def test_valid_list_passes(self):
        validate_schema_objects([good()], GoodChunk, "id")

    def test_empty_list_passes(self):
        validate_schema_objects([], GoodChunk, "id")

    def test_not_a_list_raises(self):
        with pytest.raises(ValueError, match="must return a list"):
            validate_schema_objects(good(), GoodChunk, "id")

    def test_wrong_type_at_index_zero_raises(self):
        with pytest.raises(ValueError, match="index 0"):
            validate_schema_objects([{"id": "pk1"}], GoodChunk, "id")

    def test_wrong_type_names_expected_class(self):
        with pytest.raises(ValueError, match="GoodChunk"):
            validate_schema_objects([{"id": "pk1"}], GoodChunk, "id")

    def test_wrong_type_at_later_index_raises(self):
        objs = [good("a"), good("b"), {"id": "bad"}]
        with pytest.raises(ValueError, match="index 2"):
            validate_schema_objects(objs, GoodChunk, "id")

    def test_different_schema_class_raises(self):
        with pytest.raises(ValueError, match="OtherChunk"):
            validate_schema_objects([OtherChunk("pk1", "text", [0.1])], GoodChunk, "id")

    def test_none_primary_key_raises(self):
        with pytest.raises(ValueError, match="None"):
            validate_schema_objects([good(pk=None)], GoodChunk, "id")

    def test_primary_key_field_absent_from_class_raises(self):
        # User declared PRIMARY_KEY = "doc_id" but their dataclass only has "id"
        with pytest.raises(ValueError, match="doc_id"):
            validate_schema_objects([good()], GoodChunk, "doc_id")
