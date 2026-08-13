#!/usr/bin/env python3
"""Generate the website's measurement vocabulary from `hif schema`.

`src/lib/measurements.ts` in the ai-interpretability repo is the site's single
source for keys, names, units, definitions and subjects — everything else over
there reads from it. It was a hand transcription of `hif schema`, and it did
what hand transcriptions do: the hif-v4 cut took the CLI from sixteen rows to
six and the site went on rendering all sixteen, pinned at `hif-v3.1`.

So it is generated now, the same way docs/FLAGS.md is:

    python3 tools/gen_site_measurements.py            # write the file
    python3 tools/gen_site_measurements.py --check    # exit 1 if it has drifted

`--check` runs in tools/hooks/pre-commit alongside sync_docs, so a change to
the registry cannot silently leave the site describing the old set.

WHAT IS GENERATED AND WHAT IS NOT
---------------------------------
Every field the CLI owns — key, name, unit, definition, observable,
functional, resolution, subject, subject_under_surrogate, surrogate_group,
acquisition — is copied verbatim from `hif schema`. Nothing here paraphrases
the CLI.

The `reading` field is the site's own plain-language gloss and is NOT the
CLI's. It lives in `src/lib/measurement-readings.json` in the site repo,
because it is site prose and belongs in the site's history. This generator
only joins the two.

That join is where the guarantee lives: a reading for a key `hif schema` does
not carry is an error (it would be a gloss on nothing), and a schema key with
no reading is an error (the site would render a measurement it cannot
explain). Both fail here rather than at runtime.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from site_paths import site_repo

REPO = Path(__file__).resolve().parent.parent
# Not `REPO.parent / …` — see tools/site_paths.py for why that misses in a
# git worktree, and misses quietly.
DEFAULT_SITE = site_repo()
REL_OUT = Path("src/lib/measurements.ts")
REL_READINGS = Path("src/lib/measurement-readings.json")

BANNER = """// ─────────────────────────────────────────────────────────────────────────────
// measurements.ts — GENERATED. Do not edit by hand.
//
//   cd <hif repo> && python3 tools/gen_site_measurements.py
//
// Every field below except `reading` is copied verbatim from `hif schema`,
// the CLI that produces every number on this site. `reading` is the site's
// own gloss and is authored in measurement-readings.json beside this file.
//
// This file was a hand transcription until hif-v4. The cut took the CLI from
// sixteen measurements to six; the transcription kept rendering sixteen,
// pinned at hif-v3.1, for as long as nobody re-read it. Generating it is the
// fix, and tools/hooks/pre-commit fails on drift.
// ─────────────────────────────────────────────────────────────────────────────
"""


def ts_string(value: str) -> str:
    """Emit a TypeScript single-quoted literal.

    json.dumps gives a double-quoted JS-compatible literal with correct
    escaping; convert to single quotes so the file matches the surrounding
    style, escaping any bare single quote first.
    """
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def schema() -> dict:
    out = subprocess.run(
        [sys.executable, "-m", "hif.cli", "schema"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def render(doc: dict, readings: dict[str, str]) -> str:
    measurements = doc["measurements"]

    missing = sorted(set(measurements) - set(readings))
    if missing:
        raise SystemExit(
            f"gen_site_measurements: no reading for {missing}.\n"
            f"Add one to {REL_READINGS} — the site cannot render a measurement "
            f"it has no gloss for."
        )
    orphan = sorted(set(readings) - set(measurements))
    if orphan:
        raise SystemExit(
            f"gen_site_measurements: readings for keys `hif schema` does not "
            f"carry: {orphan}.\nRemove them from {REL_READINGS} — a gloss on a "
            f"retired measurement has nothing to gloss."
        )

    lines = [BANNER]
    lines.append(f"export const HIF_SCHEMA_VERSION = {ts_string(doc['schema_version'])};")
    lines.append(
        f"export const HIF_SIGNAL_SET_VERSION = {ts_string(doc['signal_set_version'])};\n"
    )

    resolutions = list(doc["resolutions"])
    subjects = list(doc["subjects"])
    acquisitions = list(doc.get("acquisitions", {}))

    lines.append(
        "export type Resolution = " + " | ".join(ts_string(r) for r in resolutions) + ";"
    )
    lines.append(
        "export type Subject = " + " | ".join(ts_string(s) for s in subjects) + ";"
    )
    lines.append(
        "export type Acquisition = " + " | ".join(ts_string(a) for a in acquisitions) + ";\n"
    )

    for const, label, source in (
        ("RESOLUTIONS", "Resolution", doc["resolutions"]),
        ("SUBJECTS", "Subject", doc["subjects"]),
        ("ACQUISITIONS", "Acquisition", doc.get("acquisitions", {})),
    ):
        lines.append(f"/** `hif schema` → `{const.lower()}`. */")
        lines.append(f"export const {const}: Record<{label}, string> = {{")
        for key, text in source.items():
            lines.append(f"  {ts_string(key)}: {ts_string(text)},")
        lines.append("};\n")

    lines.append(
        "/** True when the number does not describe the profiled model. */\n"
        "export function isTargetSubject(s: Subject): boolean {\n"
        "  return s === 'target-distribution' || s === 'target-output-text';\n"
        "}\n"
    )

    lines.append("export interface Measurement {")
    lines.append("  key: string;")
    lines.append("  name: string;")
    lines.append("  unit: string;")
    lines.append("  /** The CLI's own definition, verbatim. */")
    lines.append("  definition: string;")
    lines.append("  /** The site's plain-language gloss (measurement-readings.json). */")
    lines.append("  reading: string;")
    lines.append("  observable: string;")
    lines.append("  functional: string;")
    lines.append("  resolution: Resolution;")
    lines.append("  subject: Subject;")
    lines.append("  /** Subject when a surrogate stood in for the target. Null = unchanged. */")
    lines.append("  subjectUnderSurrogate: Subject | null;")
    lines.append("  surrogateGroup: string | null;")
    lines.append("  /** What taking the measurement brings into existence. */")
    lines.append("  acquisition: Acquisition;")
    lines.append("}\n")

    lines.append("export const MEASUREMENTS: Record<string, Measurement> = {")
    for key in sorted(measurements):
        row = measurements[key]
        sus = row.get("subject_under_surrogate")
        sg = row.get("surrogate_group")
        lines.append(f"  {key}: {{")
        lines.append(f"    key: {ts_string(key)},")
        lines.append(f"    name: {ts_string(row['name'])},")
        lines.append(f"    unit: {ts_string(row['unit'])},")
        lines.append(f"    definition: {ts_string(row['definition'])},")
        lines.append(f"    reading: {ts_string(readings[key])},")
        lines.append(f"    observable: {ts_string(row['observable'])},")
        lines.append(f"    functional: {ts_string(row['functional'])},")
        lines.append(f"    resolution: {ts_string(row['resolution'])},")
        lines.append(f"    subject: {ts_string(row['subject'])},")
        lines.append(
            f"    subjectUnderSurrogate: {ts_string(sus) if sus else 'null'},"
        )
        lines.append(f"    surrogateGroup: {ts_string(sg) if sg else 'null'},")
        lines.append(f"    acquisition: {ts_string(row['acquisition'])},")
        lines.append("  },")
    lines.append("};\n")

    lines.append("/** Every key `hif schema` carries. */")
    lines.append(
        "export const MEASUREMENT_KEYS: string[] = Object.keys(MEASUREMENTS);\n"
    )

    # Accessors over the data above. Registry-independent, so they are emitted
    # as static text rather than derived — they read MEASUREMENTS and do not
    # change when a row is added or cut.
    lines.append(ACCESSORS)
    return "\n".join(lines)


ACCESSORS = """export function measurement(key: string): Measurement | undefined {
  return MEASUREMENTS[key];
}

/**
 * What the site shows for a measurement: the CLI's own name for the quantity.
 *
 * This used to prefer a coined shorthand and fall back to `name`, which meant
 * a reader met "Stability" on the panel and `input_entropy_std_bits` on the
 * chip below it and had to be told they were the same thing. There is one name
 * now, and it is the one the CLI prints.
 */
export function displayLabel(key: string): string {
  return MEASUREMENTS[key]?.name ?? key;
}

/** One-line "whose data is this" note, or null when it is the target's own. */
export function subjectNote(key: string): string | null {
  const m = MEASUREMENTS[key];
  if (!m) return null;
  if (m.subject === 'prompt-only') {
    return `Prompt-only — ${SUBJECTS['prompt-only']}`;
  }
  if (m.subject === 'target-output-text' || m.subject === 'mixed') {
    return `${m.subject} — ${SUBJECTS[m.subject]}`;
  }
  if (m.subjectUnderSurrogate) {
    return `Under a surrogate this becomes ${m.subjectUnderSurrogate} — ${SUBJECTS[m.subjectUnderSurrogate]}`;
  }
  return null;
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report drift and exit 1 instead of writing.")
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE,
                        help=f"The site repo (default: {DEFAULT_SITE}).")
    args = parser.parse_args()

    if not args.site.is_dir():
        print(f"gen_site_measurements: {args.site} not present — nothing to do.")
        return 0

    readings_path = args.site / REL_READINGS
    if not readings_path.is_file():
        print(f"gen_site_measurements: missing {readings_path}", file=sys.stderr)
        return 2

    text = render(schema(), json.loads(readings_path.read_text()))
    dest = args.site / REL_OUT

    if dest.is_file() and dest.read_text() == text:
        if args.check:
            print("gen_site_measurements: site vocabulary is current.")
        return 0

    if args.check:
        print(f"gen_site_measurements: {dest} has drifted from `hif schema`.",
              file=sys.stderr)
        print("\nRun: python3 tools/gen_site_measurements.py", file=sys.stderr)
        return 1

    dest.write_text(text)
    print(f"gen_site_measurements: wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
