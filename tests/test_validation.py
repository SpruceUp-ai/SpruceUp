"""Tests for validate_schema_objects and validate_pipeline."""

import types

import pytest
from dataclasses import dataclass

from spruceup.config import SpruceUpConfig
from spruceup.utils.validation import validate_pipeline, validate_schema_objects


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


def _config(transform=lambda: None) -> SpruceUpConfig:
    # validate_pipeline only inspects type(config) and config.transform, so the
    # connector fields can be placeholders here.
    return SpruceUpConfig(sources=[], target=None, embedder=None, transform=transform)


class TestValidatePipeline:

    def test_valid_pipeline_passes(self):
        pipeline = types.SimpleNamespace(config=_config())
        validate_pipeline(pipeline)  # no raise

    def test_missing_config_attribute_exits(self):
        pipeline = types.SimpleNamespace()  # no `config` at all
        with pytest.raises(SystemExit, match="config is not defined"):
            validate_pipeline(pipeline)

    def test_config_set_to_none_exits(self):
        pipeline = types.SimpleNamespace(config=None)
        with pytest.raises(SystemExit, match="config is not defined"):
            validate_pipeline(pipeline)

    def test_config_wrong_type_exits(self):
        pipeline = types.SimpleNamespace(config={"not": "a config"})
        with pytest.raises(SystemExit, match="must be the result of defineConfig"):
            validate_pipeline(pipeline)

    def test_config_without_transform_exits(self):
        pipeline = types.SimpleNamespace(config=_config(transform=None))
        with pytest.raises(SystemExit, match="no transform function"):
            validate_pipeline(pipeline)

    def test_error_message_names_the_pipeline_file(self):
        pipeline = types.SimpleNamespace(config=None)
        with pytest.raises(SystemExit, match="spruceup_pipeline.py is misconfigured"):
            validate_pipeline(pipeline)
