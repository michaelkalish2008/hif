#!/usr/bin/env python3
"""Count a published corpus by SUBJECT — whose behaviour each number describes.

A measurement key appearing in a record does not mean the number is about the
model named on it. `subject` answers that, and it degrades per run: a backend
that cannot teacher-force has its input-side rows recovered from the *prompt* by
a reference model, and a backend returning only the selected token has its
output-side rows recovered by teacher-forcing a proxy over the text the target
actually emitted. The first is not about the target at all; the second is about
its text rather than its distributions.

This walks a corpus of profile JSON and reports the split, reading the subject
rule from `hif/profile/registry.py` rather than restating it — so the table
cannot drift from what the CLI would say about the same run.

    python3 tools/corpus_subjects.py ../ai-interpretability/public/data

Counts are per profile, averaged over the regimes each model was run on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hif.profile.registry import MEASUREMENT_REGISTRY, effective_subject  # noqa: E402

# Subjects, ordered from "about the model itself" to "not about it at all".
COLUMNS = [
    ("target-distribution", "own distributions"),
    ("target-output-text", "readings of its text"),
    ("mixed", "mixed"),
    ("prompt-only", "prompt-only"),
]


def subjects_for_run(record: dict) -> dict[str, str]:
    """key -> effective subject, from the surrogates this run actually used."""
    findings = record.get("findings") or {}
    return {
        m.key: effective_subject(
            m,
            input_surrogate=bool(findings.get("surrogate_model_name")),
            output_surrogate=bool(findings.get("output_distribution_surrogate_name")),
        )
        for m in MEASUREMENT_REGISTRY
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path, help="directory of <model>/<regime>.json")
    args = ap.parse_args()

    files = sorted(args.corpus.glob("*/*.json"))
    if not files:
        print(f"no profiles under {args.corpus}", file=sys.stderr)
        return 2

    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_profiles: dict[str, int] = defaultdict(int)

    for f in files:
        model = f.parent.name
        record = json.loads(f.read_text())
        subjects = subjects_for_run(record)
        n_profiles[model] += 1
        # A quantity is counted where the record actually put it. `measurements`
        # carries what the run claims about the target; `prompt_measurements`
        # carries what a reference model said about the prompt, which is
        # reported separately precisely because it is about something else.
        for key in record.get("measurements") or {}:
            tally[model][subjects.get(key, "unregistered")] += 1
        for key in record.get("prompt_measurements") or {}:
            tally[model]["prompt-only"] += 1

    order = sorted(tally, key=lambda m: (-tally[m]["target-distribution"] / n_profiles[m], m))
    present = [(k, label) for k, label in COLUMNS if any(tally[m][k] for m in tally)]
    width = max(len(m) for m in tally)

    print(f"{'model':<{width}}" + "".join(f"{label:>22}" for _, label in present))
    print("-" * (width + 22 * len(present)))
    for model in order:
        n = n_profiles[model]
        row = "".join(f"{tally[model][k] / n:>22.2f}" for k, _ in present)
        print(f"{model:<{width}}{row}")
    print(f"\nper profile, averaged over the regimes each model was run on "
          f"({len(files)} profiles, {len(tally)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
