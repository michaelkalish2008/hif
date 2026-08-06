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

from hif.cli import app
from hif.cli._compat import _artifact_signal_set_version, _signal_set_family
from hif.profile.registry import SIGNAL_SET_VERSION
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
    # The current set is hif-v4: the cut-to-core REMOVED ten keys, and a
    # removal is always a family break — intersecting across it would silently
    # treat "we no longer claim this" as "both runs measured this". (v3 broke
    # from v2 over the prompt-only split; v2 from v1 over normalized/levels.)
    # Pin the FAMILY, not the exact version — this test's own closing
    # assertion is that a minor bump must not orphan artifacts, so pinning
    # the exact string would forbid the very thing it exists to allow.
    assert _signal_set_family(SIGNAL_SET_VERSION) == "hif-v4"
    assert _signal_set_family(SIGNAL_SET_VERSION) != _signal_set_family("hif-v3.4")
    assert _signal_set_family(SIGNAL_SET_VERSION) != _signal_set_family("hif-v2")
    # The whole point of the family rule: a future minor bump must not orphan
    # artifacts stamped with the current version.
    assert _signal_set_family(SIGNAL_SET_VERSION + ".1") == _signal_set_family(
        SIGNAL_SET_VERSION
    )
    # The bump that admitted Shift ◆ (output_step_jsd_bits) and its top-K
    # overlap companion is the concrete instance of that rule: purely additive,
    # so a hif-v3 artifact and a hif-v3.1 one still intersect.
    assert _signal_set_family("hif-v3.1") == _signal_set_family("hif-v3")


def test_the_v4_cut_is_exactly_the_declared_core():
    """A removal is a major bump, and the surviving set is pinned by name.

    The predecessor of this test asserted the v3 keys were all still present,
    with a failure message that read "that is a MAJOR bump" — and it fired for
    exactly that reason when hif-v4 cut ten rows. The set is now pinned
    exactly: a row silently vanishing OR silently returning both fail, because
    each cut row fell to evidence recorded in the SIGNAL_SET_VERSION history
    and readmission must argue with that evidence, not drift past it.
    """
    from hif.profile.signals import MEASUREMENT_KEYS

    core = {
        "input_entropy_shift_bits", "input_entropy_std_bits",
        "perturbation_jsd_bits", "io_cosine_similarity",
        "prompt_surprisal_excess_bits", "output_entropy_bits",
    }
    assert set(MEASUREMENT_KEYS) == core, (
        "the measurement set no longer matches the declared hif-v4 core — "
        f"unexpected: {set(MEASUREMENT_KEYS) ^ core}"
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

    monkeypatch.setattr(cli._app.console, "width", 100_000)  # no soft-wrap in JSON
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
