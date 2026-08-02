"""When two profile artifacts may be compared at all.

`hif compare` reports differences, never a verdict — but two artifacts can be
incomparable in a way no delta could express: a different modality is a
different experimental condition, and a different major signal-set family
means the two runs did not measure the same set. Both are hard errors here
rather than caveats on the table.
"""

from __future__ import annotations

import re

import typer

from hif.cli_base import err_console


def _profile_modality(p) -> str:
    return getattr(p.prompt, "modality", "text") or "text"


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


def _modality_mismatch_exit(baseline_modality: str, candidate_modality: str) -> None:
    """Cross-modality comparison is a different experimental condition, not a
    difference in the model — hard error, exit 2."""
    err_console.print(
        f"[red]A {baseline_modality} profile is a different experimental "
        f"condition than a {candidate_modality} profile. Re-profile both "
        "under the same modality to compare.[/red]"
    )
    raise typer.Exit(2)
