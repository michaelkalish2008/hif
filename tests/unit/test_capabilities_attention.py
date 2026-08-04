"""The attention capability class is empty since hif-v4, and must stay so.

The two attention-row measurements were cut: their numbers come from a fixed
bidirectional encoder reading text, the input-side row was bit-identical across
all fifteen corpus models by construction, and a measurement set for models
should not carry a quantity that is never about one.

ATTENTION_METRICS is derived from the registry (`observable == "attention row"`)
rather than hand-listed, so this file's single assertion is the whole guard: if
a row with that observable is ever readmitted, the set becomes non-empty and
this fails — forcing the readmission to argue with the hif-v4 evidence in the
SIGNAL_SET_VERSION history instead of drifting past it.
"""

from __future__ import annotations

from hif.models.capabilities import ATTENTION_METRICS


def test_no_attention_row_measurements_are_registered():
    assert ATTENTION_METRICS == frozenset(), (
        f"attention-row measurements reappeared: {sorted(ATTENTION_METRICS)} — "
        "see the hif-v4 history in hif/profile/registry.py before readmitting"
    )
