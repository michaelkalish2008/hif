"""The encoder loads from the local HF cache before it touches the network.

`SentenceTransformer(repo_id)` revalidates against huggingface.co on every
construction. On a machine with no route to the host that is five retries and
then a hard pipeline failure — for an encoder already sitting in
~/.cache/huggingface. The encoder is a fixed local dependency of the
measurement, so the cached copy is asked for first.
"""

from __future__ import annotations

import sys
import types

import pytest

from hif.clustering.embed import EmbeddingModel
from hif.config import EmbeddingConfig


class _FakeST:
    """Records how it was constructed; optionally refuses the offline load."""

    calls: list[dict] = []
    fail_local: bool = False

    def __init__(self, name, **kwargs):
        type(self).calls.append({"name": name, **kwargs})
        if kwargs.get("local_files_only") and type(self).fail_local:
            raise OSError("not in the local cache")
        self.name = name

    def get_sentence_embedding_dimension(self):
        return 384


@pytest.fixture
def fake_st(monkeypatch, tmp_path):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _FakeST.calls = []
    _FakeST.fail_local = False
    return _FakeST


def _config(tmp_path):
    return EmbeddingConfig(cache_dir=tmp_path / "emb")


def test_first_attempt_is_local_only(fake_st, tmp_path):
    EmbeddingModel(_config(tmp_path))

    assert len(fake_st.calls) == 1, fake_st.calls
    assert fake_st.calls[0]["local_files_only"] is True


def test_falls_back_to_the_hub_when_not_cached(fake_st, tmp_path):
    """A first run on a machine that has never downloaded it must still work."""
    fake_st.fail_local = True

    EmbeddingModel(_config(tmp_path))

    assert len(fake_st.calls) == 2, fake_st.calls
    assert fake_st.calls[0]["local_files_only"] is True
    assert "local_files_only" not in fake_st.calls[1]
