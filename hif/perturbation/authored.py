"""Researcher-authored perturbation variants, carried in the workload row format.

The generators in this package author paraphrases by rule. This module is the
opposite pole of that axis: the researcher authors every variant, the tool
authors nothing.

There is deliberately no second file format here. Case data in this project
travels as workload JSONL rows (hif/batch.py) — one JSON object per line,
`query_id` + `text` — and a variants file is the same format with one more
key:

    {"query_id": "q1", "text": "<prompt>", "variants": ["<paraphrase>", ...]}

One schema, three uses: `hif batch` profiles the rows directly (a row's
`variants` ride along), `[perturbation] variants_file` points a single
`hif profile` run at the same file, and the researcher edits one kind of
thing. JSONL also carries paraphrase text better than any delimited format —
prompts contain commas and newlines, and JSON escaping is defined where CSV
quoting is a negotiation.

Outputs are NOT written back here. Outputs are records: `--variant-io` puts a
`variant_io` block (input paraphrase → elicited continuation, per variant) in
the run's JSON record, next to `output_text` where the baseline continuation
already lives. Inputs stay immutable; there is one source of output truth.
"""

from __future__ import annotations

from pathlib import Path


class AuthoredVariantsError(ValueError):
    """The variants file cannot supply what the run needs.

    A hard error by design: an empty or mismatched file silently degrading to
    an un-perturbed run would report measurements under a config that claims
    perturbation happened."""


def load_authored_variants(path: Path, prompt: str) -> list[str]:
    """The variants the researcher wrote for this prompt, in file order.

    The file is workload JSONL (hif/batch.py schema) whose rows carry a
    `variants` list. Rows are matched by EXACT string equality on `text` — a
    normalising match (case, whitespace) would mean the file tested a
    different prompt than the run measured. Multiple matching rows
    concatenate, so variants can be grouped one-per-line or listed together.
    """
    from hif.batch import WorkloadError, load_workload

    try:
        rows = load_workload(path)
    except WorkloadError as exc:
        raise AuthoredVariantsError(str(exc))

    variants: list[str] = []
    matched = 0
    for row in rows:
        if row.text != prompt:
            continue
        matched += 1
        variants.extend(v for v in (row.variants or []) if v.strip())
    if not variants:
        raise AuthoredVariantsError(
            f"variants_file {path} has no usable variants for this prompt "
            f"({matched} row(s) matched `text` but none carried a non-empty "
            f"`variants` entry). An empty perturbation set under a perturbed "
            f"config is an error, not a default."
        )
    return variants


def variant_io_block(profile, sink: dict) -> list[dict]:
    """The record's `variant_io` block: what was sent, what came back.

    Joins the profile's perturbation entries (generator + variant inputs —
    already part of the profile) with the caller-owned sink of elicited
    continuations. A variant that was never generated from (synthesized-input
    tier, or a generation failure) carries `"output": None` — absent means
    not elicited, and inventing a value would be a fabricated elicitation.
    """
    return [
        {
            "generator": record.generator,
            "input": variant,
            "output": sink.get(variant),
        }
        for record in profile.perturbations
        for variant in record.variants
    ]
