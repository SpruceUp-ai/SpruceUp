import pytest

from spruceup.manifest import Manifest


@pytest.fixture
def manifest(tmp_path):
    m = Manifest(path=str(tmp_path / "spruceup_manifest.db"))
    try:
        yield m
    finally:
        m.close()
