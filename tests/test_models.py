import pytest

from text_summarization_project.models.registry import resolve_family


def test_resolve_known_family():
    entry = resolve_family("t5")
    assert "model_cls" in entry and "tokenizer_cls" in entry


def test_resolve_unknown_family_raises():
    with pytest.raises(KeyError):
        resolve_family("not-a-real-family")
