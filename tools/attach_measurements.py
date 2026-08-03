"""Attach the CLI's own measurement answer to a written profile artifact.

The trace artifact carries `metrics.*` — the raw computation — and no
`measurements` block. Anything reading `metrics.stability.perturbation_jsd_bits`
directly gets the number BEFORE the absence rules in hif/profile/measure.py are
applied, which is how a gpt-5 run that generated zero output steps published
`perturbation_jsd_bits = 0.0` as a measurement of the model.

The fix is not to reimplement those rules in the consumer. There are four of
them, they are subtle, and a second implementation is a second thing to drift —
which is the failure this project keeps finding. Instead the artifact carries
what `measurements()` returned, so a consumer reads the CLI's answer rather than
deriving its own.

Two blocks, and the split is load-bearing:

    measurements          quantities about the model named in the record
    prompt_measurements   quantities about the PROMPT under a reference model.
                          Identical across every model profiled on the same
                          prompt, so reading one as a property of the target is
                          a category error. See docs/MEASUREMENTS.md § Subject.

Idempotent: run it over a directory as often as you like.

    python3 tools/attach_measurements.py ../ai-interpretability/public/data
"""

import json
import sys
from pathlib import Path

from hif.profile.measure import measurements, prompt_measurements
from hif.profile.schema import BehavioralRangeProfile


def attach(path: Path) -> tuple[bool, str]:
    raw = json.loads(path.read_text())
    try:
        profile = BehavioralRangeProfile.model_validate(raw)
    except Exception as exc:  # a pre-0.10.0 artifact will not validate
        return False, f"cannot rehydrate: {type(exc).__name__}"

    raw["measurements"] = measurements(profile)
    raw["prompt_measurements"] = prompt_measurements(profile)

    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(raw, separators=(",", ":")))
    tmp.rename(path)
    return True, f"{len(raw['measurements'])} measurements, {len(raw['prompt_measurements'])} prompt-only"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    files = sorted(root.rglob("*.json"))
    if not files:
        print(f"no profiles under {root}")
        return 1

    ok = skipped = 0
    for f in files:
        done, note = attach(f)
        if done:
            ok += 1
        else:
            skipped += 1
            print(f"  skip {f.relative_to(root)}: {note}")
    print(f"attached: {ok}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
