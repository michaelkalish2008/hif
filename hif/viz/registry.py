"""The signal-visualization registry.

Single source of truth for the viz engine: which signals exist, their identity
(id/label/kind/glyph/family), the generator that draws them, and the
availability predicate that decides whether *this* profile has the backing data.

Insertion order is the v1 ordered set, so the engine renders signals in a
stable, taxonomy-faithful order:
  aggregates: stability, breadth, surprise, io_correlation, sensitivity,
              continuity, similarity
  readings:   entropy, shift, wager, spread, horizon, exposure

FIDELITY CONTRACT: one visualization per signal, gated on data availability.
No signal is ever rendered from another signal's data, and no absent signal is
drawn as zero/flat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from hif.viz.signals import (
    breadth, continuity, entropy, exposure, horizon, io_correlation,
    sensitivity, shift, similarity, spread, stability, surprise, wager,
)


@dataclass(frozen=True)
class SignalViz:
    id: str
    label: str
    kind: str        # "aggregate" | "reading"
    family: str      # "information" | "geometric"
    glyph: str | None
    generate: Callable
    available: Callable  # (profile) -> str | None


# Ordered as the v1 signal set. `module` supplies generate/available.
_SPEC = [
    # id, label, kind, family, module
    ("stability",      "Stability",       "aggregate", "information", stability),
    ("breadth",        "Breadth",         "aggregate", "information", breadth),
    ("surprise",       "Surprise",        "aggregate", "information", surprise),
    ("io_correlation", "I/O Correlation", "aggregate", "information", io_correlation),
    ("sensitivity",    "Sensitivity",     "aggregate", "information", sensitivity),
    ("continuity",     "Continuity",      "aggregate", "geometric",   continuity),
    ("similarity",     "Similarity",      "aggregate", "geometric",   similarity),
    ("entropy",        "Entropy",         "reading",   "information", entropy),
    ("shift",          "Shift",           "reading",   "information", shift),
    ("wager",          "Wager",           "reading",   "information", wager),
    ("spread",         "Spread",          "reading",   "geometric",   spread),
    ("horizon",        "Horizon",         "reading",   "geometric",   horizon),
    ("exposure",       "Exposure",        "reading",   "geometric",   exposure),
]

SIGNALS: list[SignalViz] = [
    SignalViz(
        id=sid, label=label, kind=kind, family=family,
        glyph=getattr(mod, "GLYPH", None),
        generate=mod.generate, available=mod.available,
    )
    for (sid, label, kind, family, mod) in _SPEC
]

SIGNALS_BY_ID: dict[str, SignalViz] = {s.id: s for s in SIGNALS}

AGGREGATE_SIGNALS = [s.id for s in SIGNALS if s.kind == "aggregate"]
READING_SIGNALS = [s.id for s in SIGNALS if s.kind == "reading"]
