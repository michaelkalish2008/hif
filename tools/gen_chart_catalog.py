"""Generate a JSON catalog of the chart set, from hif/viz/registry.py.

Run: python tools/gen_chart_catalog.py [outfile] [--model ID] [--tokens N]

The companion site shows one sample of every chart. The list of what those
charts ARE has to come from the registry that draws them — a hand-written copy
on the site would be a second source of truth for the chart set, which is the
same failure the measurement reference already had once.

Emits, per signal: the registry id (which is also the filename `--charts`
writes), the chart label, the kind, and the measurement key it joins to, if
any. Nothing here is display copy; the site supplies its own framing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hif.viz.registry import SIGNALS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "docs" / "chart-catalog.json"

# Signals whose backing data is opt-in. `--charts` alone renders these as
# "not available for this run" placeholders; they need the analyzers that
# --diagnostics turns on. Derived from the availability predicates in
# hif/viz/signals/, which the registry does not expose as data.
NEEDS_DIAGNOSTICS = {"spread", "horizon"}


def build(sample_model: str | None = None, sample_tokens: int | None = None) -> dict:
    return {
        "generated_by": "tools/gen_chart_catalog.py",
        "command": 'hif profile <model> "<prompt>" --charts --output-dir out',
        # Which run the shipped sample charts came from. Recorded here rather
        # than written into the site's copy, so the provenance travels with the
        # catalog and cannot drift from the files it describes.
        "sample": {"model": sample_model, "max_new_tokens": sample_tokens},
        "signals": [
            {
                "id": s.id,
                "label": s.label,
                "kind": s.kind,
                "measurement_key": getattr(s, "measurement_key", None),
                "needs_diagnostics": s.id in NEEDS_DIAGNOSTICS,
                "file": f"{s.id}.html",
            }
            # Aggregates first, then readings — the order the dashboard uses.
            for s in sorted(SIGNALS, key=lambda x: (x.kind != "aggregate", x.id))
        ],
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    model = tokens = None
    positional = []
    while args:
        a = args.pop(0)
        if a == "--model":
            model = args.pop(0)
        elif a == "--tokens":
            tokens = int(args.pop(0))
        else:
            positional.append(a)
    out = Path(positional[0]) if positional else DEFAULT_OUT
    catalog = build(model, tokens)
    out.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"wrote {out} ({len(catalog['signals'])} signals, sample={model})")
