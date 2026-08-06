"""When two profile artifacts may be compared at all.

`hif compare` reports differences, never a verdict — but two artifacts can be
incomparable in a way no delta could express: a different signal-set family
means the two runs did not claim the same measurement set, and intersecting
across that would silently read "we no longer claim this" as "both runs
measured this". A hard error here rather than a caveat on the table.

(A second gate, cross-modality comparison, was removed with the image path in
hif-v4 — with one modality there is nothing to mismatch.)
"""

from __future__ import annotations

import re

import typer

from hif.cli._app import err_console


def _signal_set_family(version: str) -> str:
    """Major family of a signal-set version: "hif-v1.1" -> "hif-v1".

    Versions within one family are additive supersets: comparison proceeds
    over the intersection of measurements present in both artifacts, naming
    each exclusion. Different families are a true mismatch.
    """
    m = re.match(r"^(.*-v\d+)", version or "")
    return m.group(1) if m else (version or "")


def _artifact_signal_set_version(data: dict) -> str:
    """Signal-set version recorded on a profile/baseline/prior JSON dict.

    Priors and baselines record `protocol_version`; hosted profiles record
    `signal_set_version`. Artifacts predating both read as "hif-v1"."""
    return data.get("signal_set_version") or data.get("protocol_version") or "hif-v1"


def _signal_set_mismatch_exit(baseline_version: str, candidate_version: str) -> None:
    """Different major signal-set families: hard error, exit 2 (mirrors the
    platform 409). Same-family minor differences never reach here — they
    compare over the intersection instead."""
    err_console.print(
        f"[red]These artifacts were scored under different signal sets "
        f'("{baseline_version}" vs "{candidate_version}"). Re-profile them '
        "under the same HIF Signal Set version to compare.[/red]"
    )
    raise typer.Exit(2)

