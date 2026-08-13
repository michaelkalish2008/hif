"""Bring a published profile artifact into agreement with the current absence rules.

A profile written before a given absence rule existed carries the number that
rule now withholds, and nothing in the file says so. `attach_measurements.py`
solves half of this: it refreshes the derived `measurements` block so a consumer
reads the CLI's answer instead of deriving its own. But the block is derived,
and the *stored* fields underneath it are not — a fabricated zero written into
`output_side.mean_step_entropy` stays a zero however many times the blocks are
recomputed.

This tool repairs those stored fields. The rule it applies is exactly one thing:

    make each stored value equal what the current code would produce
    from this artifact's own data

Nothing is invented and nothing is recomputed from the model — every input is
already in the file. In practice that means nulling the quantities that the
hif-v4.2 empty-generation pass made `Optional`, on the runs where the target
generated nothing:

    output_side.mean_step_entropy              mean over no steps
    center.output_mean_entropy                 same
    center.entropy_ratio                       a ratio needing that term
    center.prompt_output_cosine_distance       0.0 is the MINIMUM of [0, 2],
                                               so on a distance it claimed the
                                               output was identical to the
                                               prompt — for a model that
                                               returned nothing
    metrics.sensitivity[].output_entropy_delta a difference of two absent means

and recomputing the trend slope from the stored per-step series, which is
`None` below two points rather than a flat 0.0:

    metrics.similarity.trend
    findings.similarity_trend_slope

`schema_version` is deliberately NOT bumped. It records which hif version ran,
which is still true — the run happened when it happened, and claiming 0.13.0
for an artifact nobody re-ran would be the same kind of false statement this
tool exists to remove. What changed is recorded in `notes` instead, so a reader
can tell a repaired file from a fresh one.

Idempotent, and safe on artifacts that need nothing.

    python3 tools/repair_absences.py ../ai-interpretability/public/data
"""

import collections
import json
import re
import sys
from pathlib import Path

from hif.metrics.similarity import _similarity_trend
from hif.metrics.semantic import SemanticMetrics
from hif.profile.measure import measurements, prompt_measurements
from hif.profile.schema import BehavioralRangeProfile

# The clamp `compute_step_sensitivity` used to apply to an infinite KL. Matched
# with >= rather than == because the aggregate stored a MEAN of sentinels
# (9.65e8 on a 58-step trace where 56 were undefined), so the exact value
# varies while the magnitude does not. A real KL in bits over a top-K
# distribution cannot approach this: the ceiling is log2(1/p_min) for the
# smallest recorded probability, tens of bits at the very most.
_KL_SENTINEL = 1e8


def _differs(a, b) -> bool:
    """Whether a stored value needs replacing, tolerating float round-trip."""
    if a is None or b is None:
        return a is not b
    return abs(a - b) > 1e-12


def _sensitivity_groups(raw: dict):
    """Every place a profile stores SensitivityMetrics.

    Two, and they are the same records: `metrics.sensitivity` is the flat list
    the aggregate reduces, and `perturbations[].sensitivity` groups them by
    generator. Repairing one and not the other would leave the artifact
    disagreeing with itself.
    """
    yield (raw.get("metrics") or {}).get("sensitivity") or []
    for p in raw.get("perturbations") or []:
        yield p.get("sensitivity") or []


NOTE = (
    "Absences repaired by tools/repair_absences.py: fields that a zero-output "
    "run had published as measured zeros are now null. No value was "
    "recomputed from a model; the run itself is unchanged."
)


def _trend_from_stored(raw: dict):
    """The trend slope the current code would fit through this file's series.

    `_similarity_trend` reads `SemanticMetrics.mean_pairwise_distance` per step,
    and the artifact stores that series — so this is a recomputation from the
    file's own data, not a re-run. None below two points.
    """
    rows = (raw.get("metrics") or {}).get("semantic") or []
    try:
        parsed = [SemanticMetrics.model_validate(r) for r in rows]
    except Exception:  # noqa: BLE001 — a shape we cannot read is left alone
        return None
    return _similarity_trend(parsed)


def repair(path: Path) -> tuple[bool, str]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "output_side" not in raw:
        return False, "not a profile artifact"

    before = json.dumps(raw, sort_keys=True)
    changed: list[str] = []

    def _null(container: dict, key: str, label: str) -> None:
        if container.get(key) is not None:
            container[key] = None
            changed.append(label)

    no_output = not (raw["output_side"].get("steps") or [])

    # --- per-variant sensitivity records (schema 0.14.0) ---------------------
    # Repaired for EVERY profile, not only the zero-output ones: a variant can
    # align no steps against a perfectly healthy baseline, and the 1e9 KL
    # sentinel is independent of the baseline entirely.
    for group in _sensitivity_groups(raw):
        for i, s in enumerate(group):
            steps = s.get("step_sensitivities") or []
            for ss in steps:
                # The sentinel is finite, which is why the filter above it
                # never fired. Undefined is null.
                if isinstance(ss.get("kl_divergence"), (int, float)) and ss["kl_divergence"] >= _KL_SENTINEL:
                    ss["kl_divergence"] = None
                    changed.append("step_sensitivities[].kl_divergence (1e9 sentinel)")
            if s.get("n_steps_aligned") != len(steps):
                s["n_steps_aligned"] = len(steps)
                changed.append(f"sensitivity[{i}].n_steps_aligned")
            n_undef = sum(1 for ss in steps if ss.get("kl_divergence") is None)
            if s.get("n_undefined_kl_steps") != n_undef:
                s["n_undefined_kl_steps"] = n_undef
                changed.append(f"sensitivity[{i}].n_undefined_kl_steps")
            # Recompute each mean from the stored per-step series, which is
            # what the current code would produce from this file's own data.
            for field, key in (
                ("mean_js_divergence", "js_divergence"),
                ("mean_kl_divergence", "kl_divergence"),
                ("mean_entropy_delta", "entropy_delta"),
                ("mean_nucleus_stability_p90", "nucleus_overlap_p90"),
            ):
                vals = [ss.get(key) for ss in steps]
                present = [v for v in vals if v is not None]
                new = sum(present) / len(present) if present else None
                if _differs(s.get(field), new):
                    s[field] = new
                    changed.append(f"sensitivity[{i}].{field}")

    stability = (raw.get("metrics") or {}).get("stability")
    if stability is not None:
        sens = (raw.get("metrics") or {}).get("sensitivity") or []
        aligned = sum(1 for s in sens if s.get("mean_js_divergence") is not None)
        if stability.get("n_perturbations_aligned") != aligned:
            stability["n_perturbations_aligned"] = aligned
            changed.append("stability.n_perturbations_aligned")
        js = [s.get("mean_js_divergence") for s in sens]
        js = [v for v in js if v is not None]
        new_jsd = sum(js) / len(js) if js else None
        if sens and _differs(stability.get("perturbation_jsd_bits"), new_jsd):
            stability["perturbation_jsd_bits"] = new_jsd
            changed.append("stability.perturbation_jsd_bits")

    if no_output:
        _null(raw["output_side"], "mean_step_entropy", "output_side.mean_step_entropy")
        center = raw.get("center") or {}
        for key in (
            "output_mean_entropy",
            "entropy_ratio",
            "prompt_output_cosine_distance",
        ):
            _null(center, key, f"center.{key}")
        for i, s in enumerate((raw.get("metrics") or {}).get("sensitivity") or []):
            if s.get("output_entropy_delta") is not None:
                s["output_entropy_delta"] = None
                changed.append(f"metrics.sensitivity[{i}].output_entropy_delta")
        # The same records are duplicated under `perturbations[].sensitivity`.
        for p in raw.get("perturbations") or []:
            for s in p.get("sensitivity") or []:
                if s.get("output_entropy_delta") is not None:
                    s["output_entropy_delta"] = None
                    changed.append("perturbations[].sensitivity[].output_entropy_delta")

    # The trend is recomputed rather than nulled: it is undefined below two
    # steps on ANY run, not only an empty one, so the stored series decides.
    trend = _trend_from_stored(raw)
    sim = (raw.get("metrics") or {}).get("similarity")
    if sim is not None and sim.get("trend") != trend:
        sim["trend"] = trend
        changed.append("metrics.similarity.trend")
    findings = raw.get("findings") or {}
    if findings.get("similarity_trend_slope") != trend:
        findings["similarity_trend_slope"] = trend
        changed.append("findings.similarity_trend_slope")

    if not changed:
        # Still refresh the derived blocks — cheap, idempotent, and the whole
        # point is that the file agrees with the current rules.
        _refresh_blocks(raw)
        if json.dumps(raw, sort_keys=True) == before:
            return False, "already current"
        _write(path, raw)
        return True, "measurement blocks refreshed"

    note = (raw.get("notes") or "").strip()
    if NOTE not in note:
        raw["notes"] = f"{note}\n\n{NOTE}".strip()

    _refresh_blocks(raw)
    _write(path, raw)
    # Collapsed by field, not enumerated: a profile with 15 variants x 6
    # repaired fields prints 90 near-identical lines otherwise, and the one
    # that matters scrolls away.
    kinds = collections.Counter(re.sub(r"\[\d+\]", "[]", c) for c in changed)
    summary = ", ".join(
        f"{k} x{n}" if n > 1 else k for k, n in sorted(kinds.items())
    )
    return True, f"{len(changed)} field(s): {summary}"


def _refresh_blocks(raw: dict) -> None:
    """Recompute `measurements` / `prompt_measurements` under the current rules.

    Same computation as tools/attach_measurements.py, run here so one command
    leaves the file wholly consistent rather than half-repaired. A file that
    cannot be rehydrated keeps whatever blocks it had — refusing to guess is the
    same rule the blocks themselves follow.

    The two blocks are written DIFFERENTLY when empty, and the asymmetry is
    load-bearing in both directions:

    `measurements` stays `{}`. It is not "no block"; it is a block that says
    this run published no measurement, which is exactly what a zero-output run
    has to say. It also has to be present: the site's resolver reads
    `if (profile.measurements) return profile.measurements[id] ?? null` and
    otherwise falls back to deriving the value straight off `metrics.*` — where
    `similarity.io_sim` still holds the 0.17 this whole repair is about. An
    absent block would hand the fabricated number back to the page.

    `prompt_measurements` becomes absent (`None`), following
    `measure.prompt_measurement_block`: an empty block there would assert that
    the run considered those quantities and found nothing, when in fact none
    was in play.
    """
    try:
        profile = BehavioralRangeProfile.model_validate(raw)
    except Exception:  # noqa: BLE001
        return
    raw["measurements"] = measurements(profile)
    prompt_only = prompt_measurements(profile)
    raw["prompt_measurements"] = prompt_only or None


def _write(path: Path, raw: dict) -> None:
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(raw, separators=(",", ":")))
    tmp.rename(path)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    files = sorted(root.rglob("*.json"))
    if not files:
        print(f"no profiles under {root}")
        return 1

    repaired = untouched = 0
    for f in files:
        done, note = repair(f)
        if done:
            repaired += 1
            print(f"  {f.relative_to(root)}: {note}")
        else:
            untouched += 1
    print(f"repaired: {repaired}, unchanged: {untouched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
