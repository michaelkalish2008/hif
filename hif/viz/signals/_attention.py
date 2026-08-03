"""Shared extraction for the attention-row readings, output side and input side.

Both read the stored ``profile.attention_capture`` (a TextAttentionAnalysis)
and reduce an aggregated attention matrix to a per-token row-entropy trace.
The matrix is head/layer-aggregated in the stored profile, so this is a
faithful *aggregated* reading of attention diffuseness; we report what the
profile stored and label it as such. Returns None when the data isn't present.
"""

from __future__ import annotations

import numpy as np


def row_entropy_trace(weights: list[list[float]]) -> list[float] | None:
    """Per-row Shannon entropy, in bits, over each row's causal prefix.

    Row i is restricted to columns 0..i (causal), renormalised so the row is a
    distribution, and its entropy computed.

    There is deliberately no ``normalized`` option. The historical Horizon
    reading divided each value by ``log2(prefix_len)`` to land in [0, 1]; that
    denominator is the sequence length, so it leaks position metadata into a
    number presented as behaviour — the same mistake as the removed
    vocabulary-size normaliser (see hif/profile/registry.py). Bits only.
    """
    if not weights:
        return None
    out: list[float] = []
    for i, row in enumerate(weights):
        prefix = np.asarray(row[: i + 1], dtype=np.float64)
        s = prefix.sum()
        if s <= 0 or len(prefix) == 0:
            out.append(0.0)
            continue
        p = prefix / s
        p = p[p > 1e-12]
        out.append(float(-np.sum(p * np.log2(p))))
    return out


def get_attention_map(profile, which: str):
    """Return (tokens, weights) for the requested map, or (None, None).

    which = "input"  → prompt self-attention (Horizon)
    which = "output" → continuation attention (Spread)
    """
    att = getattr(profile, "attention_capture", None)
    if att is None:
        att = getattr(profile, "attention", None)  # pre-rename fallback
    if att is None:
        return None, None
    # profile.attention_capture is typed Any: a TextAttentionAnalysis in-memory,
    # a dict when loaded from JSON. Coerce so both paths work.
    if isinstance(att, dict):
        try:
            from hif.analysis.attention import TextAttentionAnalysis
            att = TextAttentionAnalysis.model_validate(att)
        except Exception:  # noqa: BLE001
            return None, None
    try:
        if which == "input":
            amap = att.input_analysis.attention_map
        else:
            amap = att.continuation_attention
        return list(amap.tokens), list(amap.weights)
    except Exception:  # noqa: BLE001 — any structural mismatch → treat as absent
        return None, None
