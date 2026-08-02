"""The attention rows are gated on the analysis STAGE, never on the backend.

`hif/analysis/attention.py` loads its own bidirectional encoder
(distilbert-base-uncased) and reads TEXT: the prompt for
attention_entropy_input_bits, the target's generated continuation for
attention_entropy_output_bits. The target model's attention is never accessed —
the module says so itself. So neither measurement needs anything from the
backend beyond text, which every backend returns.

`hif/models/capabilities.py` used to claim the opposite in its docstring ("only
open HuggingFace models expose attention") and enforce that claim in
`metric_support`, while correctly stating a few lines later that the attention
"is not the target's". The gate told users their backend could not produce a
measurement it produces perfectly well. These tests hold the corrected gate in
place.
"""

from __future__ import annotations

import pytest

from hif.models.capabilities import (
    ATTENTION_METRICS,
    BACKENDS,
    metric_support,
    signals_available,
)

BACKEND_NAMES = sorted(BACKENDS)


@pytest.mark.parametrize("metric", sorted(ATTENTION_METRICS))
@pytest.mark.parametrize("backend", BACKEND_NAMES)
def test_every_backend_can_produce_the_attention_rows(metric, backend):
    """No backend refusal — the encoder reads text, and all of them return it."""
    assert metric_support(metric, backend) is None
    assert metric_support(metric, backend, attention_enabled=True) is None


@pytest.mark.parametrize("metric", sorted(ATTENTION_METRICS))
@pytest.mark.parametrize("backend", BACKEND_NAMES)
def test_the_real_requirement_is_the_stage_and_the_message_says_so(metric, backend):
    reason = metric_support(metric, backend, attention_enabled=False)
    assert reason is not None
    assert "--diagnostics" in reason
    # The fix must not send the user to a different backend: switching backends
    # would not help, and telling them it would is the original false claim.
    assert "--backend hf" not in reason


def test_attention_availability_does_not_vary_across_backends():
    """The invariant the old gate broke: this column was never backend-shaped."""
    rows = {
        name: tuple(
            signals_available(name)[m] for m in sorted(ATTENTION_METRICS)
        )
        for name in BACKEND_NAMES
    }
    assert len(set(rows.values())) == 1, rows
    assert set(rows[BACKEND_NAMES[0]]) == {True}


def test_backend_info_no_longer_carries_an_attention_capability():
    """The field encoded a capability nothing in the pipeline ever asked for.

    Leaving it (set to True everywhere) would still imply a per-backend
    question; removing it is the honest correction.
    """
    for info in BACKENDS.values():
        assert not hasattr(info, "attention")


def test_the_teacher_forcing_gate_is_untouched():
    """The real backend dependence still holds — this is not a blanket widening."""
    assert metric_support("prompt_surprisal_excess_bits", "hf") is None
    reason = metric_support("prompt_surprisal_excess_bits", "anthropic")
    assert reason is not None and "teacher" in reason
