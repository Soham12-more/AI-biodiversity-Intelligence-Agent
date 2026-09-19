import os

import pytest

os.environ["DISABLE_GEO"] = "1"
os.environ["DISABLE_LLM"] = "1"
os.environ["USE_EMBEDDINGS"] = "0"


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    from app import store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
