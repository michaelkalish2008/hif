#!/usr/bin/env python3
"""Check that the published corpus describes the current measurement set.

The website's corpus is 120 profiles of ~5 MB each, stored in Vercel Blob.
Nothing tied it to this repo: hif-v4 cut ten measurements and the published
profiles went on carrying `measurements` blocks with seven retired keys in
them, unstamped, until someone thought to look. The docs have a drift check
(tools/sync_docs.py) and so does the site's vocabulary
(tools/gen_site_measurements.py). This is the third.

    python3 tools/check_corpus.py --write ../ai-interpretability/public/data
    python3 tools/check_corpus.py --check
    python3 tools/check_corpus.py --check --deep

WHY A MANIFEST
--------------
390 MB is not a thing a pre-commit hook can fetch, so `--write` scans the
local corpus and emits a small `manifest.json` that ships beside it. `--check`
fetches that one file and compares it against MEASUREMENT_REGISTRY.

The manifest is DERIVED, never asserted: every key in it was read out of a
profile on disk. It is still only as current as the run that wrote it, which
is why `--write` belongs immediately before the upload and `--deep` exists —
`--deep` pulls real profiles and verifies they match what the manifest claims,
so a manifest that has fallen behind its own corpus is caught rather than
believed. The hook runs the cheap check; run `--deep` after any corpus push.

WHAT COUNTS AS DRIFT
--------------------
A published key that no longer exists in the registry, or a corpus whose
`signal_set_version` is from a different FAMILY than this one. NOT drift: a
profile carrying fewer keys than the registry defines — that is the absence
rules working, and it is the normal case for every backend that cannot
teacher-force. A check that demanded all six everywhere would fail on two
thirds of the corpus and be deleted within a week, which is worse than no
check.

Also NOT drift: a minor version behind, e.g. a hif-v4 corpus against a
hif-v4.1 repo. This compared version strings exactly until hif-v4.1, which was
correct only because every bump until then had been major. A minor bump is
additive by definition — `hif compare` intersects across it, and the
`_signal_set_family` rule exists to say so — so a corpus one minor behind
publishes keys that all still mean what they meant. Demanding an exact match
would have required recomputing 120 profiles to add a measurement none of them
took, and the real hazard, a published key the registry has retired, is caught
by the check below regardless of version.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hif.cli._compat import _signal_set_family  # noqa: E402
from hif.profile.registry import MEASUREMENT_BY_KEY, SIGNAL_SET_VERSION  # noqa: E402

DEFAULT_CORPUS = REPO.parent / "ai-interpretability" / "public" / "data"
PUBLISHED_BASE = "https://ai-interpretability.com/data"
MANIFEST_NAME = "manifest.json"
TIMEOUT = 30


def scan(corpus: Path) -> dict:
    """Build the manifest by reading every profile. Nothing here is asserted."""
    profiles = sorted(
        p for p in corpus.glob("*/*.json")
        if not p.name.endswith("_attention.json") and p.name != MANIFEST_NAME
    )
    if not profiles:
        raise SystemExit(f"check_corpus: no profiles under {corpus}")

    models: dict[str, dict] = {}
    versions: set[str] = set()
    all_keys: set[str] = set()

    for path in profiles:
        doc = json.loads(path.read_text())
        measured = sorted(doc.get("measurements") or {})
        prompt = sorted(doc.get("prompt_measurements") or {})
        all_keys.update(measured, prompt)
        versions.add(str(doc.get("signal_set_version")))
        entry = models.setdefault(
            path.parent.name, {"regimes": [], "measurements": set(), "prompt_measurements": set()}
        )
        entry["regimes"].append(path.stem)
        entry["measurements"].update(measured)
        entry["prompt_measurements"].update(prompt)

    return {
        "signal_set_version": sorted(versions)[0] if len(versions) == 1 else sorted(versions),
        "n_profiles": len(profiles),
        "keys_published": sorted(all_keys),
        "models": {
            name: {
                "regimes": sorted(e["regimes"]),
                "measurements": sorted(e["measurements"]),
                "prompt_measurements": sorted(e["prompt_measurements"]),
            }
            for name, e in sorted(models.items())
        },
    }


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def verify(manifest: dict) -> list[str]:
    """Return the reasons this manifest disagrees with the registry."""
    problems: list[str] = []

    version = manifest.get("signal_set_version")
    if _signal_set_family(str(version)) != _signal_set_family(SIGNAL_SET_VERSION):
        problems.append(
            f"corpus signal_set_version is {version!r}, this repo is "
            f"{SIGNAL_SET_VERSION!r} — a different family, so the sets do "
            f"not intersect"
        )

    retired = sorted(set(manifest.get("keys_published") or []) - set(MEASUREMENT_BY_KEY))
    if retired:
        problems.append(
            "published profiles carry keys the registry no longer defines: "
            + ", ".join(retired)
        )
    return problems


def deep_check(manifest: dict, base: str, per_model: int) -> list[str]:
    """Pull real profiles and confirm the manifest is not describing a past corpus."""
    problems: list[str] = []
    for model, entry in manifest.get("models", {}).items():
        for regime in entry["regimes"][:per_model]:
            url = f"{base}/{model}/{regime}.json"
            try:
                doc = fetch(url)
            except Exception as exc:  # noqa: BLE001 — reported, not raised
                problems.append(f"{model}/{regime}: fetch failed ({exc})")
                continue
            keys = set(doc.get("measurements") or {}) | set(doc.get("prompt_measurements") or {})
            unlisted = sorted(keys - set(manifest.get("keys_published") or []))
            if unlisted:
                problems.append(f"{model}/{regime}: carries {unlisted}, absent from the manifest")
            if doc.get("signal_set_version") != SIGNAL_SET_VERSION:
                problems.append(
                    f"{model}/{regime}: signal_set_version is "
                    f"{doc.get('signal_set_version')!r}"
                )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", type=Path, nargs="?", const=DEFAULT_CORPUS, metavar="CORPUS",
                    help=f"Scan a local corpus and write its manifest (default: {DEFAULT_CORPUS}).")
    ap.add_argument("--check", action="store_true",
                    help="Fetch the published manifest and compare it to the registry.")
    ap.add_argument("--deep", action="store_true",
                    help="With --check: also pull real profiles to catch a stale manifest.")
    ap.add_argument("--per-model", type=int, default=1,
                    help="Profiles per model to pull under --deep (default 1).")
    ap.add_argument("--base", default=PUBLISHED_BASE, help="Published corpus base URL.")
    args = ap.parse_args()

    if args.write is not None:
        corpus = args.write
        if not corpus.is_dir():
            print(f"check_corpus: {corpus} not present — nothing to scan.")
            return 0
        manifest = scan(corpus)
        dest = corpus / MANIFEST_NAME
        dest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"check_corpus: wrote {dest} — {manifest['n_profiles']} profiles, "
            f"signal_set_version={manifest['signal_set_version']}, "
            f"{len(manifest['keys_published'])} distinct keys"
        )
        problems = verify(manifest)
        if problems:
            print("\ncheck_corpus: the corpus you just scanned disagrees with this repo:",
                  file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        return 0

    if not args.check:
        ap.print_help()
        return 2

    try:
        manifest = fetch(f"{args.base}/{MANIFEST_NAME}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("check_corpus: no published manifest — run --write and upload it.",
                  file=sys.stderr)
            return 1
        print(f"check_corpus: could not reach the corpus ({exc}) — skipping.")
        return 0
    except Exception as exc:  # noqa: BLE001
        # Offline is not drift. A hook must not block a commit on the network.
        print(f"check_corpus: could not reach the corpus ({exc}) — skipping.")
        return 0

    problems = verify(manifest)
    if args.deep:
        problems += deep_check(manifest, args.base, args.per_model)

    if problems:
        print("check_corpus: the published corpus disagrees with this repo:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nRecompute the measurement blocks and re-upload, then rerun --write.\n"
            "The corpus is the evidence for every number on the site; a corpus\n"
            "describing a retired set is a published claim this repo cannot make.",
            file=sys.stderr,
        )
        return 1

    print(
        f"check_corpus: corpus is current — {manifest.get('n_profiles')} profiles at "
        f"{manifest.get('signal_set_version')}"
        + (", deep-checked" if args.deep else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
