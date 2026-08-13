"""The docs' backend lists must be the registry's, not a copy of it.

Two access-tier tables are published — README.md's and docs/MEASUREMENTS.md's
"Backend Access" — and the site renders them on the same page as the flag
reference, so a stale one is not merely wrong, it visibly contradicts the
others. The README's is generated (tools/gen_backend_tiers.py); this asserts
it has been regenerated. MEASUREMENTS.md's keeps hand-written per-tier prose
that nothing can introspect, so its BACKEND NAMES are checked instead —
membership is the part that drifted, and it is the part the registry owns.

The README table omitted `ollama`, which is a backend, and credited `[F]` with
attention, which is not a backend capability at all — the kind of thing a
reader cannot catch and a test can.

What these tests deliberately do NOT assert is that every backend-shaped name
in the README is a registry key. That table also named `deepseek`, which is
not a `--backend` value but IS a supported target behind `--backend openai`
with a `base_url`. A subset-of-BACKENDS check over the prose would fail on
that sentence and pressure someone into deleting a true fact to make a test
pass. The generated block is checked exactly; the prose around it is not.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import re
from pathlib import Path

import pytest

from hif.models.capabilities import BACKENDS

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "docs" / "MEASUREMENTS.md"


def _generator():
    """Import tools/gen_backend_tiers.py — a script, not an installed module."""
    path = ROOT / "tools" / "gen_backend_tiers.py"
    spec = importlib.util.spec_from_file_location("gen_backend_tiers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_tier_table_is_regenerated():
    gen = _generator()
    current = gen.README.read_text()
    assert gen._replace(current, gen.build()) == current, (
        "README.md's backend access-tier table has drifted from "
        "hif/models/capabilities.py. Run: python3 tools/gen_backend_tiers.py"
    )


def test_every_backend_appears_in_the_readme_table():
    """Not just 'no invented names' — no backend may be left out either.

    A subset check would have passed the version that omitted `ollama`, which
    is the failure a reader actually pays for: a real backend that the doc
    says nothing about reads as one that does not exist.
    """
    gen = _generator()
    listed = {n for names in gen.tier_members().values() for n in names}
    assert listed == set(BACKENDS)


def test_a_new_access_tier_stops_the_generator():
    """An unmapped `logprobs` value must fail loudly, not silently drop rows.

    The tier tags and their one-line descriptions are the only hand-written
    claims in generated output, so they are the only ones the generator cannot
    check. Falling through would publish a table missing a backend while still
    reading as the complete list — the exact failure being fixed.
    """
    gen = _generator()
    invented = dataclasses.replace(BACKENDS["hf"], name="future", logprobs="rank-only")
    gen.BACKENDS = {**BACKENDS, "future": invented}
    with pytest.raises(SystemExit, match="rank-only"):
        gen.tier_members()


@pytest.mark.parametrize("tag", ["[F]", "[T-k]", "[P]"])
def test_measurements_access_table_matches_the_registry(tag):
    """docs/MEASUREMENTS.md names exactly this tier's backends, and no others."""
    gen = _generator()
    expected = set(gen.tier_members()[tag])

    row = next(
        (
            line for line in MEASUREMENTS.read_text().splitlines()
            if line.startswith(f"| `{tag}`")
        ),
        None,
    )
    assert row is not None, (
        f"docs/MEASUREMENTS.md § Backend Access has no `{tag}` row. If the tier "
        "vocabulary moved, move it here and in tools/gen_backend_tiers.py "
        "together — it is also used by AGENTS.md and mock_backends.py."
    )
    # Column 2 is the backend list. Later columns name MEASUREMENT keys in
    # backticks too, so the scan is scoped to that cell rather than the row.
    cell = row.split("|")[2]
    assert set(re.findall(r"`([^`]+)`", cell)) == expected, (
        f"docs/MEASUREMENTS.md's `{tag}` row disagrees with BACKENDS. "
        f"Expected: {sorted(expected)}"
    )
