"""Tests for the application archetype registry (hif/archetypes)."""

from dataclasses import fields

import pytest

from hif.archetypes import (
    Archetype,
    UnknownArchetypeError,
    _parse_flat_yaml,
    list_archetypes,
    load_archetype,
)

EXPECTED_IDS = [
    "agent-tool-use",
    "classification",
    "coding-assistant",
    "document-understanding",
    "extraction",
    "multimodal-qa",
    "rag-qa",
    "summarization",
    "support-chatbot",
]

# Multimodal archetypes carry the smaller default analysis window.
MULTIMODAL_IDS = {"multimodal-qa", "document-understanding"}


# ---------------------------------------------------------------------------
# list_archetypes
# ---------------------------------------------------------------------------


def test_list_archetypes_returns_all_sorted():
    assert list_archetypes() == EXPECTED_IDS


# ---------------------------------------------------------------------------
# load_archetype — valid loads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("archetype_id", EXPECTED_IDS)
def test_load_archetype_valid(archetype_id):
    a = load_archetype(archetype_id)
    assert isinstance(a, Archetype)
    assert a.id == archetype_id
    assert a.description
    if archetype_id in MULTIMODAL_IDS:
        assert a.default_analysis_window == 256
    assert a.default_analysis_window == "adaptive" or isinstance(
        a.default_analysis_window, int
    )


def test_archetype_carries_no_reference_prior():
    """An archetype supplies a label and a default analysis window. It never
    names a stored reference profile to compare against — an archetype must
    not decide what a measurement is judged by."""
    a = load_archetype("agent-tool-use")
    assert not hasattr(a, "prior")
    assert "prior" not in {f.name for f in fields(Archetype)}


def test_agent_tool_use_uses_adaptive_window():
    assert load_archetype("agent-tool-use").default_analysis_window == "adaptive"


def test_coding_assistant_uses_adaptive_window():
    assert load_archetype("coding-assistant").default_analysis_window == "adaptive"


def test_support_chatbot_has_integer_window():
    assert load_archetype("support-chatbot").default_analysis_window == 512


def test_archetype_carries_no_unconsumed_fields():
    """Regression for the dead --application surface: perturbation_family and
    report_template were loaded and validated as required but consumed
    nowhere, so the flag's help overstated what it did. The registry must not
    reacquire fields the pipeline does not read."""
    a = load_archetype("rag-qa")
    assert not hasattr(a, "perturbation_family")
    assert not hasattr(a, "report_template")


# ---------------------------------------------------------------------------
# load_archetype — unknown id
# ---------------------------------------------------------------------------


def test_load_archetype_unknown_id_raises():
    with pytest.raises(UnknownArchetypeError) as excinfo:
        load_archetype("does-not-exist")
    assert excinfo.value.archetype_id == "does-not-exist"
    assert excinfo.value.valid_ids == EXPECTED_IDS


# ---------------------------------------------------------------------------
# flat yaml parser
# ---------------------------------------------------------------------------


def test_parse_flat_yaml_types():
    data = _parse_flat_yaml(
        "# comment\n"
        "id: foo\n"
        "window: 512\n"
        "adaptive: adaptive\n"
        "optional: null\n"
        "quoted: 'a: b'\n"
        "flag: true\n"
    )
    assert data == {
        "id": "foo",
        "window": 512,
        "adaptive": "adaptive",
        "optional": None,
        "quoted": "a: b",
        "flag": True,
    }


def test_parse_flat_yaml_malformed_line_raises():
    with pytest.raises(ValueError):
        _parse_flat_yaml("just a bare line\n")
