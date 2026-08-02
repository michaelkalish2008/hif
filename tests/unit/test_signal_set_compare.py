"""Additive-superset signal-set compare semantics.

Versions within one major family (hif-v2, hif-v2.1, ...) are additive
supersets: `hif compare` proceeds over the intersection of measurements
present in both artifacts, naming each excluded measurement on stderr.
Different major families (hif-v1 vs hif-v2) are a true mismatch — hard
error, exit 2. A minor bump must never orphan artifacts stamped with an
earlier minor of the same family.
"""

import json

from typer.testing import CliRunner

from hif.cli import (
    SIGNAL_SET_VERSION,
    _artifact_signal_set_version,
    _signal_set_family,
    app,
)
from tests.unit.profile_helpers import _make_profile

runner = CliRunner()


# ---------------------------------------------------------------------------
# Family parsing + artifact version reading
# ---------------------------------------------------------------------------


def test_signal_set_family_extracts_major():
    assert _signal_set_family("hif-v1") == "hif-v1"
    assert _signal_set_family("hif-v1.1") == "hif-v1"
    assert _signal_set_family("hif-v1.12") == "hif-v1"
    assert _signal_set_family("hif-v2") == "hif-v2"
    assert _signal_set_family("hif-v2.3") == "hif-v2"
    # Unrecognized shapes are their own family
    assert _signal_set_family("custom") == "custom"
    assert _signal_set_family("") == ""


def test_current_version_has_a_wellformed_family():
    # The current set is hif-v3: moving the prompt-only quantities out of
    # `measurements` REMOVES keys from the set, so it is deliberately not in
    # the hif-v2 family — intersecting a hif-v2 artifact with a hif-v3 one
    # would silently compare a fact about the target against a fact about a
    # reference model. (hif-v2 itself broke from hif-v1 over normalized/levels.)
    # Pin the FAMILY, not the exact version — this test's own closing
    # assertion is that a minor bump must not orphan artifacts, so pinning
    # the exact string would forbid the very thing it exists to allow.
    assert _signal_set_family(SIGNAL_SET_VERSION) == "hif-v3"
    assert _signal_set_family(SIGNAL_SET_VERSION) != _signal_set_family("hif-v2")
    assert _signal_set_family(SIGNAL_SET_VERSION) != _signal_set_family("hif-v1")
    # The whole point of the family rule: a future minor bump must not orphan
    # artifacts stamped with the current version.
    assert _signal_set_family(SIGNAL_SET_VERSION + ".1") == _signal_set_family(
        SIGNAL_SET_VERSION
    )


def test_artifact_version_reads_both_fields_and_defaults():
    assert _artifact_signal_set_version({"protocol_version": "hif-v1"}) == "hif-v1"
    assert _artifact_signal_set_version({"signal_set_version": "hif-v1.1"}) == "hif-v1.1"
    # signal_set_version wins when both present
    assert (
        _artifact_signal_set_version(
            {"signal_set_version": "hif-v1.1", "protocol_version": "hif-v1"}
        )
        == "hif-v1.1"
    )
    # Artifacts predating either field read as hif-v1
    assert _artifact_signal_set_version({}) == "hif-v1"


# ---------------------------------------------------------------------------
# hif compare
# ---------------------------------------------------------------------------


def _write_profile(path, version=None):
    raw = json.loads(_make_profile().model_dump_json())
    if version is not None:
        raw["signal_set_version"] = version
    path.write_text(json.dumps(raw))
    return raw


def test_compare_same_versions_no_notice(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_profile(a)
    _write_profile(b)
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert "Signal sets differ" not in result.output
    assert "different signal sets" not in result.output


def test_compare_minor_version_difference_proceeds(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_profile(a, version="hif-v2")
    _write_profile(b, version="hif-v2.1")
    result = runner.invoke(app, ["compare", str(a), str(b)])
    # Same family: comparison proceeds, no family hard error, real output.
    assert result.exit_code == 0
    assert "different signal sets" not in result.output
    assert "Profile comparison" in " ".join(result.output.split())


def test_compare_major_family_mismatch_exits_2(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_profile(a, version="hif-v1.1")
    _write_profile(b, version="hif-v2")  # normalized/levels removal: breaking
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 2
    flat = result.output.replace("\n", " ")
    assert "different signal sets" in flat
    assert "Re-profile them under the same HIF Signal Set version" in flat


def test_compare_json_delta_covers_shared_measurements_only(tmp_path, monkeypatch):
    import hif.cli as cli

    monkeypatch.setattr(cli.console, "width", 100_000)  # no soft-wrap in JSON
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_profile(a)
    _write_profile(b)
    result = runner.invoke(app, ["compare", str(a), str(b), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert set(data["delta"]) == set(data["measurements_a"]) & set(
        data["measurements_b"]
    )
    # Every delta carries the unit it is measured in — no bare, unitless
    # numbers, and no normalized/level block anywhere in the payload.
    assert set(data["units"]) == set(data["delta"])
    assert "normalized" not in data and "levels" not in data
